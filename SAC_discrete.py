"""
2023.05.09 made by soh 

- from SAC.py

- Found that it works well for CartPole-v1 completing training after about 50 episodes,
    even alpha = 0.
    This means the power comes from DQN-architecture at least in this simple environment. 

"""
import sys
import json
#   from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from replaybuffer import ReplayBuffer, PERBuffer
from explorer import Explorer_replayBufferFiller as Explorer

class SAC_discrete:
    def __init__(self, mode, config, logger, observDim, actionDim):
        self.config = config
        self.logger = logger
        self.mode = mode  # config["Mode"]
        self.npDtype = np.dtype(config["dtype"])
        self.tfDtype = tf.convert_to_tensor(np.zeros((1,), dtype=self.npDtype)).dtype  # tf doesn't have dtype()
        self.npIntDtype = np.dtype(config["intdtype"])
        self.actionDim = actionDim

        self.isTargetActor = config["TargetActor"]
        self.isCritic2 = config["Critic2"]
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

        if self.isRewardNorm:
            # self.recentMemoryCapacity = self.memoryCapacity // 4
            # if "train" in self.mode:
            #     self.reward_memory = deque(maxlen=self.memoryCapacity // 2)
            #     self.recent_reward = deque(maxlen=self.recentMemoryCapacity)

            self.reward_norm_steps = 200
            self.reward_mean = 1
            # self.rewardNormalizationThreshold = 0.1  # begin reward normalization after replay memory is filled over threshold
            self.rewardNormalizationThreshold = 0.7

        self.batchSz = config["BatchSize"]
        self.tau = config["SoftUpdateRate_tau"] 
        self.actor_lr = config["Actor_learningRate"]
        self.critic_lr = config["Critic_learningRate"]
        self.gamma = tf.Variable(config["RewardDiscountRate_gamma"], dtype=self.tfDtype) 
        self.alpha = tf.Variable(config["TemperatureParameter_alpha"], dtype=self.tfDtype)
        self.eps = 1e-6  # tiny quentity added to prevent inf; NOTE: value < 1e-6 (like 1e-7) is considered as 0 causing inf
        self.logStd_min = -13  # e**(-13) = 2.26e-06; for stds
        self.logStd_max = 1

        self.explorer = Explorer(mode, config, self.savePath, self.replayBuffer)

        self.batchNormInUnitsList = config["BatchNorm_inUnitsList"]  # to represent batchNorm in XXX_units list like [64,'bn',64]
        actor_hiddenUnits = config["Actor_hiddenUnits"]     # like [64, 'bn', 64], 'bn' for BatchNorm
        critic_hiddenUnits = config["Critic_hiddenUnits"]   # like [64, 'bn', 64], 'bn' for BatchNorm

        if mode == "train":
            self.actor = self.build_actor(observDim, actor_hiddenUnits, actionDim, self.tfDtype)
            self.critic1 = self.build_critic(observDim, critic_hiddenUnits, actionDim, self.tfDtype)
            self.target_critic1 = self.build_critic(
                    observDim, critic_hiddenUnits, actionDim, self.tfDtype, trainable=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = self.build_actor(
                        observDim, actor_hiddenUnits, actionDim, self.tfDtype, trainable=False)
            if self.isCritic2:
                self.critic2 = self.build_critic(observDim, critic_hiddenUnits, actionDim, self.tfDtype)
                self.target_critic2 = self.build_critic(
                        observDim, critic_hiddenUnits, actionDim, self.tfDtype, trainable=False)
                self.critic2_optimizer = Adam(self.critic_lr)
        elif mode == "test": 
            self.actor = load_model(f"{self.savePath}/actor/")
            self.logger.info(f"actor is loaded from {self.savePath}/actor/")
            self.actor.summary(print_fn=self.logger.info)
        elif mode == "continued_train":
            self.actor = load_model(f"{self.savePath}/actor/")
            self.critic1 = load_model(f"{self.savePath}/critic1/")
            self.target_critic1 = load_model(f"{self.savePath}/target_critic1/")
            if self.isCritic2:
                self.critic2 = load_model(f"{self.savePath}/critic2/")
                self.target_critic2 = load_model(f"{self.savePath}/target_critic2/")
            if self.isTargetActor:
                self.target_actor = load_model(f"{self.savePath}/target_actor/")
            self.actor.summary(print_fn=self.logger.info)
            self.critic1.summary(print_fn=self.logger.info)
            self.explorer.load()

    def build_actor(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        observ = Input(shape=(observDim,), dtype=dtype, name="actor_in")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"actor_hidden_{ix}")(h)
        actionProb = self.dense_or_batchNorm(actionDim, "softmax", trainable=trainable, name="actor_probability")(h)

        net = Model(inputs=observ, outputs=actionProb, name="actor")
        return net

    def get_actionProb_logActionProb(self, observ, withTarget=False):
        prob = self.target_actor(observ) if withTarget else self.actor(observ)  # (batchSz,actionDim)
        self.logger.debug(f"in get_actionProb_logActionProb:prob={prob}")
        logProb = tf.math.log(prob + self.eps)                              # (batchSz,actionDim)
        return prob, logProb

    def build_critic(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        observ = Input(shape=(observDim,), dtype=dtype, name="critic_in")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "tanh", trainable=trainable, name=f"hidden_{ix}")(h)
        action = self.dense_or_batchNorm(actionDim, "linear", use_bias=False, trainable=trainable, name="Qs")(h)

        net = Model(inputs=observ, outputs=action, name="critic")
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
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            actionProb, logProb = self.get_actionProb_logActionProb(observ)     # each (batchSz,actionDim)
            Q1 = self.critic1(observ)                                           # (batchSz,actionDim)
            if self.isCritic2:
                Q2 = self.critic2(observ)
                Q_min = tf.minimum(Q1, Q2)
                Q_soft = Q_min - self.alpha * logProb                           # (batchSz,actionDim)
            else:
                Q_soft = Q1 - self.alpha * logProb                              # (batchSz,actionDim)
            V_soft = tf.reduce_sum(Q_soft * actionProb, axis=1, keepdims=True)  # (batchSz,1) 
            actor_loss = -tf.reduce_mean(V_soft)                                # ()

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def update_critic(self, observ, action, reward, next_observ, done, importance_weights):
        """ Args:
            observ, next_observ: shape=(batchSz,observDim) 
            action: index of actionToEnv in possibleValuesFor[]; shape=(batchSz,actionDim) 
            done, reward: shape=(batchSz,1)
        """
        with tf.GradientTape(persistent=True) as tape:
            next_actionProb, next_logProb = self.get_actionProb_logActionProb(
                    next_observ, withTarget=self.isTargetActor) # each (batchSz,actionDim)
            target_Q1 = self.target_critic1(next_observ)                                # (batchSz,actionDim)
            if self.isCritic2:
                target_Q2 = self.target_critic2(next_observ)
                target_Q_min = tf.minimum(target_Q1, target_Q2)
                target_Q_soft = target_Q_min - self.alpha * next_logProb                # (batchSz,actionDim)
                target_V_soft = tf.reduce_sum(target_Q_soft * next_actionProb, axis=1, keepdims=True) # (batchSz,1) 
                y = reward + (1.0 - done) * self.gamma * target_V_soft                  # (batchSz,1)
                Q2 = self.critic2(observ)
                Q2_selectedByAction = tf.reduce_sum(Q2 * action, axis=1, keepdims=True) # action as mask; (batchSz,1)
                td_error2 = tf.square(y - Q2_selectedByAction)
                td_error2 = importance_weights * td_error2 if self.isPER else td_error2
                critic2_loss = tf.reduce_mean(td_error2)
            else:
                target_Q_soft = target_Q1 - self.alpha * next_logProb                   # (batchSz,actionDim)
                target_V_soft = tf.reduce_sum(target_Q_soft * next_actionProb, axis=1, keepdims=True) # (batchSz,1) 
                y = reward + (1.0 - done) * self.gamma * target_V_soft                  # (batchSz,1)
            Q1 = self.critic1(observ)                                                   # (batchSz,actionDim)
            Q1_selectedByAction = tf.reduce_sum(Q1 * action, axis=1, keepdims=True)     # action as mask; (batchSz,1)
            td_error1 = tf.square(y - Q1_selectedByAction)                              # (batchSz,1)
            td_error1 = importance_weights * td_error1 if self.isPER else td_error1
            critic1_loss = tf.reduce_mean(td_error1)                                    # ()

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        if self.isCritic2:
            critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
                #   td_error = tf.minimum(td_error1, td_error2)             # (batchSz,1)
            return critic1_loss, td_error1, critic2_loss
        else:
            return critic1_loss, td_error1

    #   @tf.function  # NOTE: recommended not to use tf.function. zip problem?? And not much enhancement. 
    def soft_update(self):
        source1 = self.critic1.variables
        target1 = self.target_critic1.variables
        for target_param1, param1 in zip(target1, source1):
            target_param1.assign(target_param1 * (1.0 - self.tau) + param1 * self.tau)
        if self.isCritic2:
            source2 = self.critic2.variables
            target2 = self.target_critic2.variables
            for target_param2, param2 in zip(target2, source2):
                target_param2.assign(target_param2 * (1.0 - self.tau) + param2 * self.tau)
        if self.isTargetActor:
            source0 = self.actor.variables
            target0 = self.target_actor.variables  
            for target_param0, param0 in zip(target0, source0):
                target_param0.assign(target_param0 * (1.0 - self.tau) + param0 * self.tau)

    #   @tf.function  # NOTE recommended not to use tf.function. cross function problem?? And not much enhancement.
    def train(self, batch, importance_weights):
        loss_error = self.update_critic(
                batch.observ,
                batch.action,
                batch.reward,
                batch.next_observ,
                batch.done,
                importance_weights
        )
        actor_loss = self.update_actor(batch.observ)
        self.soft_update()

        if self.isCritic2:
            critic1_loss, td_error1, critic2_loss = loss_error
        else:
            critic1_loss, td_error1 = loss_error
        return (critic1_loss, actor_loss), td_error1  # (,) to be consistent with DQN return

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: shape=(observDim)
        return:
            action: shape=(observDim)
        """
        if self.explorer.isReadyToExplore():
            actionToEnv = actionCoder.random_decoded()
            action = actionCoder.encode(actionToEnv)
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)                     # (1,observDim) to input to net
            actionProb, _ = self.get_actionProb_logActionProb(observ)   # (batchSz,actionDim)
            actionProb = actionProb[0]                                  # (actionDim)
            prob = actionProb.numpy()
            prob[0] = prob[0] + 1 - np.sum(prob)  # for case when tiny residue appears; to make sum(prob)=1
            idx = np.random.choice(self.actionDim, p=prob)              # index of actionToEnv; ()
            action = np.array([1 if i == idx else 0 for i in range(self.actionDim)])  # one-hot; (actionDim)
        return action

    def isReadyToTrain(self):
        b1 = self.mode == "train"
        b2 = self.replayBuffer.memoryCnt > self.memoryCnt_toStartTrain
        b3 = self.replayBuffer.memoryCnt > self.batchSz
        #   b4 = self.actCnt % 1 == 0
        return b1 and b2 and b3  #   and b4

    def save(self, msg=""):
        self.actor.save(f"{self.savePath}/actor/")
        self.critic1.save(f"{self.savePath}/critic1/")
        self.target_critic1.save(f"{self.savePath}/target_critic1/")
        if self.isTargetActor:
            self.target_actor.save(f"{self.savePath}/target_actor/")
        if self.isCritic2:
            self.critic2.save(f"{self.savePath}/critic2/")
            self.target_critic2.save(f"{self.savePath}/target_critic2/")
        self.replayBuffer.save(f"{self.savePath}/replayBuffer.json")
        self.explorer.save()
        self.logger.info(msg)

    def summary(self):
        self.actor.summary(print_fn=self.logger.info)                   # to print in logger file
        self.critic1.summary(print_fn=self.logger.info)                 # to print in logger file
        
    def summaryWrite(self, key, value, step):
        with self.writer.as_default():
            tf.summary.scalar(key, value, step=step)

