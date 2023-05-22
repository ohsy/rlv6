"""
2023.05.02 made by soh

- Made from ./rl_tf1/dqn.py using TF2 and keras.
cf. https://keras.io/examples/rl/deep_q_network_breakout/
cf. https://github.com/KerasKorea/KEKOxTutorial/blob/master/134_Keras%20와%20Gym%20과%20함께하는%20Deep%20Q-Learning%20을%20향한%20여행.md
cf. https://levelup.gitconnected.com/dqn-from-scratch-with-tensorflow-2-eb0541151049

- Found actionDim is not 2 but 1 since it is set by nElements not by number of possible values.
    => Fixed.
- Found in update_dqn(), y is of shape=(batchSz, batchSz) after the following statement 
    since reward and done is of shape=(batchSz, 1).
    y = reward + (1.0 - done) * self.gamma * target_Q_max
    => Fixed: 
        reshape reward and done
- Move run() to gym.

- NodeCode is used.

"""
import sys
import json
#   from tqdm import tqdm
import random
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from ReplayBuffer import ReplayBuffer, PERBuffer
from Explorer import Explorer_epsilonDecay as Explorer


class DQN:
    def __init__(self, mode, config, logger, observDim, actionDim):
        self.config = config
        self.logger = logger
        self.mode = mode  # config["Mode"]
        self.npDtype = np.dtype(config["dtype"])
        self.tfDtype = tf.convert_to_tensor(np.zeros((1,), dtype=self.npDtype)).dtype  # tf doesn't have dtype()
        self.npIntDtype = np.dtype(config["intdtype"])
        self.actionDim = actionDim

        self.savePath = f"{config['SavePath']}/{self.__class__.__name__}" 
        self.writer = tf.summary.create_file_writer(config["SummaryWriterPath"])
        self.isRewardNorm = config["RewardNormalization"]
        self.isPER = config["PER"]

        if self.config["PER"] == True:
            self.replayBuffer = PERBuffer(config, self.npDtype, self.tfDtype)
        else:
            self.replayBuffer = ReplayBuffer(config, self.npDtype, self.tfDtype, self.npIntDtype)
        self.memoryCapacity = config["MemoryCapacity"]
        self.memoryCnt_toStartTrain = self.config["MemoryRatio_toStartTrain"] * self.memoryCapacity
        #   self.memoryCntToFillWithRandomAction = self.config["MemoryRatio_toFillWithRandomAction"] * self.memoryCapacity

        if self.isRewardNorm:
            # self.recentMemoryCapacity = self.memoryCapacity // 4
            # if self.mode == "train":
            #     self.reward_memory = deque(maxlen=self.memoryCapacity // 2)
            #     self.recent_reward = deque(maxlen=self.recentMemoryCapacity)

            self.reward_norm_steps = 200
            self.reward_mean = 1
            # self.rewardNormalizationThreshold = 0.1  # begin reward normalization after replay buffer is filled over threshold
            self.rewardNormalizationThreshold = 0.7

        self.batchSz = config["BatchSize"]
        self.tau = config["SoftUpdateRate_tau"] 
        self.gamma = config["RewardDiscountRate_gamma"] 
        self.lr = config["DQN_learningRate"]

        self.explorer = Explorer(mode, config, self.savePath) 

        self.batchNormInUnitsList = config["BatchNorm_inUnitsList"]  # to represent batchNorm layer in XXX_units list like [64,'bn',64]
        hiddenUnits = config["DQN_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm

        if mode == "train":
            self.dqn = self.build_dqn(observDim, hiddenUnits, actionDim, self.tfDtype)
            self.target_dqn = self.build_dqn(observDim, hiddenUnits, actionDim, self.tfDtype, trainable=False)
            self.optimizer = Adam(self.lr)
        elif mode == "test": 
            self.dqn = load_model(f"{self.savePath}/dqn/")
            self.dqn.summary(print_fn=self.logger.info)
        elif mode == "continued_train":
            self.dqn = load_model(f"{self.savePath}/dqn/")
            self.target_dqn = load_model(f"{self.savePath}/target_dqn/")
            self.optimizer = Adam(self.lr)
            self.dqn.summary(print_fn=self.logger.info)
            self.explorer.load()

    def build_dqn(self, observDim, hiddenUnits, actionDim, dtype, trainable=True): 
        observ = Input(shape=(observDim,), dtype=dtype, name="inputs")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "tanh", trainable=trainable, name=f"hidden_{ix}")(h)
        action = self.dense_or_batchNorm(actionDim, "linear", trainable=trainable, name="Qs")(h)

        net = Model(inputs=observ, outputs=action)
        return net

    def dense_or_batchNorm(self, units, activation, use_bias=True, trainable=True, name=None):
        """
        Args:
            use_bias: False may be effective when outputs are symmetric
        """
        regularizationFactor = 0.01
        if units == self.batchNormInUnitsList:
            layer = BatchNormalization(dtype = self.tfDtype)
        else:
            layer = Dense(units,
                    activation = activation,
                    use_bias = use_bias,
                    kernel_regularizer = L2(regularizationFactor),
                    bias_regularizer = L2(regularizationFactor),
                    dtype = self.tfDtype,
                    trainable = trainable,
                    name = name
            )
        return layer    

    @tf.function
    def update_dqn(self, observ, action, reward, next_observ, done, importance_weights):
        """
        Args:
            observ, next_observ: shape=(batchSz,observDim)
            action: one-hot vector of the selected action-node in output layer; shape=(batchSz,actionDim)
            reward: shape=(batchSz,1) 
            done: shape=(batchSz,1) 
        """
        self.logger.debug(f"in update_dqn:")
        self.logger.debug(f"reward={reward}")
        self.logger.debug(f"done={done}")
        with tf.GradientTape() as tape:
            target_Q = self.target_dqn(next_observ)                     # shape=(batchSz, actionDim)
            self.logger.debug(f"next_observ={next_observ}")
            self.logger.debug(f"target_Q={target_Q}")
            target_Q_max = tf.reduce_max(target_Q, axis=1, keepdims=True)          # max among actionDim Qs; shape=(batchSz,1)
            self.logger.debug(f"target_Q_max={target_Q_max}")
            self.logger.debug(f"reward={reward}")
            self.logger.debug(f"done={done}")
            y = reward + (1.0 - done) * self.gamma * target_Q_max       # shape=(batchSz,1)
            self.logger.debug(f"y={y}")

            Q = self.dqn(observ)                                        # shape=(batchSz,actionDim)
            self.logger.debug(f"Q={Q}")
            self.logger.debug(f"action={action}")
            action_th_Q = tf.reduce_sum(Q * action, axis=1, keepdims=True)      # action as mask; shape=(batchSz,1)
            self.logger.debug(f"action-th_Q={action_th_Q}", flush=True)
            td_error = tf.square(y - action_th_Q)                       # shape=(batchSz,1)
            self.logger.debug(f"td_error={td_error}")
            if self.isPER:
                loss = tf.reduce_mean(importance_weights * td_error)    # shape=()
            else:
                loss = tf.reduce_mean(td_error)                         # shape=()
                self.logger.debug(f"loss={loss}")
        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))
        return loss, td_error

    #   @tf.function  # NOTE: recommended not to use tf.function. zip problem?? And not much enhancement. 
    def soft_update(self):
        source = self.dqn.variables
        target = self.target_dqn.variables
        for target_param, param in zip(target, source):
            target_param.assign(target_param * (1.0 - self.tau) + param * self.tau)

    #   @tf.function  # NOTE recommended not to use tf.function. cross function problem?? And not much enhancement.
    def train(self, batch, importance_weights):
        loss, td_error = self.update_dqn(
                batch.observ,
                batch.action,
                batch.reward,
                batch.next_observ,
                batch.done,
                importance_weights
        )
        self.soft_update()
        return loss, td_error

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: shape=(observDim)
        return:
            action: one-hot vector of the maximum-Q-node in output layer; shape=(actionDim)
        """
        if self.explorer.isReadyToExplore():
            actionToEnv = actionCoder.random_vec()  # get a random actionToEnv
            action = actionCoder.encode(actionToEnv)
            Q_max = 0  # dummy
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)  # shape=(1,observDim) to input to net

            Qs = self.dqn(observ)  # shape=(1,actionDim) 
            maxIdx = np.argmax(Qs, axis=1)[0]  # [0] gets 0th of batch. action is index of max Q among actionDim Qs. shape=()
            action = np.array([1 if i == maxIdx else 0 for i in range(self.actionDim)])  # one-hot vector; shape=(actionDim)
            Q_max = np.amax(Qs, axis=1)[0]  # [0] gets 0th of batch. max Q among actionDim Qs. scalar
        #   return action, Q_max  # Q_max for monitoring
        return action
    
    def isReadyToTrain(self):
        b1 = self.mode == "train"
        b2 = self.replayBuffer.memoryCnt > self.memoryCnt_toStartTrain 
        b3 = self.replayBuffer.memoryCnt > self.batchSz
        #   b4 = self.actCnt % 1 == 0
        return b1 and b2 and b3  #   and b4
    
    def save(self, msg=""):
        self.dqn.save(f"{self.savePath}/dqn/")
        self.target_dqn.save(f"{self.savePath}/target_dqn/")
        self.replayBuffer.save(f"{self.savePath}/replayBuffer.json")
        self.explorer.save()
        self.logger.info(msg)

    def summary(self):
        self.dqn.summary(print_fn=self.logger.info)  # to print in logger file

    def summaryWrite(self, key, value, step):
        with self.writer.as_default():
            tf.summary.scalar(key, value, step=step)

