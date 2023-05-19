"""
2023.04.25 made by soh

- Made from ddpgInGym.py using TF2 and keras.
- Losses diverge.
- At first, comment out all of '@tf.function's
- Problem: keep diverging
=> Fixed: in soft_update(): actor.variables and critic.variables are zipped together in zip() though their lengths are different 
    => separated. 
- It seems that use_bias=False in output layer in actor makes training faster.

- 1 actor, 1 critic, 1 target actor, 1 target critic: usually, average reward in episode can be larger than -0.005 
    within about 100 episodes.
- 1 actor, 2 critic, 1 target actor, 2 target critic: usually, average reward in episode can be larger than -0.005 
    within about 100 episodes. 
- 1 actor, 2 critic, 2 target critic: usually, average reward in episode can be larger than -0.005 in about 200 episodes or never. 
    In this case, @tf.function over soft_update() seems to make a bad influence. 
    I guess that for-loop in soft_update() causes some problem.

- speed up using @tf.function (per episode during train)
    with tf2.6, from 8.5sec to 6.8sec 
    with tf2.12, from 16.5sec to 6.8sec 

- Problem: tf2.12 problem with @tf.function: keep diverging
    (1) avg_reward(per episode) stays away from 0
    (2) negative avg_actor_loss(per episode) appears
    => try @tf.function only on update_actor() 
        => (1) solved and converges with 7 or 8sec per episode, but (2) is not solved.
    => I guess that multiple sequential models in critic cause problem for AutoGraph 
    => try functional model for critic => not working
=> Fixed: found batch sampling from replay buffer is included in train(). 
    Might be that this sampling is done only once. 
    Batch sample is excluded. And it converges.

- Problem: actor or critic.summary() can't show layers' output shapes.
=> Found: when the actor and critic is subclass of keras.Model, Input is not included and this problem occurs.
=> Fixed: make model_summary() where a temporary Model with Input is made and its summary() is called.

- In making Actor and Critic, functional method is chosen instead of subclassing 
    because of summary() problem. 
    I guess this problem implies subclassing keras.Model does not match with TF properly.

- isTargetActor and isCritic2 are added to be able to select the architecture of Agent.

- Agent_ddpg is changed into DDPG, which is used in game.py
"""
import sys
import json
#   from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from ReplayBuffer import ReplayBuffer, PERBuffer

class DDPG:
    def __init__(self, mode, config, logger, observDim, actionDim):
        self.config = config
        self.logger = logger
        self.mode = mode  # config["Mode"]
        self.npDtype = np.dtype(config["dtype"])
        self.tfDtype = tf.convert_to_tensor(np.zeros((1,), dtype=self.npDtype)).dtype  # tf doesn't have dtype()
        self.npIntDtype = np.dtype(config["intdtype"])

        self.isTargetActor = config["TargetActor"]
        self.isCritic2 = config["Critic2"]
        self.savePath = config["SavePath"] 
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

        self.batchNormInUnitsList = config["BatchNorm_inUnitsList"]  # to represent batchNorm layer in XXX_units list like [64,'bn',64]
        actor_hiddenUnits = config["Actor_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm
        observ_hiddenUnits = config["Critic_observationBlock_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm
        action_hiddenUnits = config["Critic_actionBlock_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm
        concat_hiddenUnits = config["Critic_concatenateBlock_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm

        if mode == "continued_train":
            self.actor = load_model(f"{self.savePath}/{self.__class__.__name__}actor/")
            self.critic1 = load_model(f"{self.savePath}/{self.__class__.__name__}critic1/")
            self.target_critic1 = load_model(f"{self.savePath}/{self.__class__.__name__}target_critic1/")
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isCritic2:
                self.critic2 = load_model(f"{self.savePath}/{self.__class__.__name__}critic2/")
                self.target_critic2 = load_model(f"{self.savePath}/{self.__class__.__name__}target_critic2/")
                self.critic2_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = load_model(f"{self.savePath}/{self.__class__.__name__}target_actor/")
            self.actor.summary(print_fn=self.logger.info)
            self.critic1.summary(print_fn=self.logger.info)
        elif mode == "train":
            self.actor = self.build_actor(observDim, actor_hiddenUnits, actionDim, self.tfDtype)
            self.critic1 = self.build_critic(observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype)
            self.target_critic1 = self.build_critic(
                    observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype, trainable=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = self.build_actor(observDim, actor_hiddenUnits, actionDim, self.tfDtype, trainable=False)
            if self.isCritic2:
                self.critic2 = self.build_critic(observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype)
                self.target_critic2 = self.build_critic(
                        observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype, trainable=False)
                self.critic2_optimizer = Adam(self.critic_lr)
        elif mode == "test": 
            self.actor = load_model(f"{self.savePath}/{self.__class__.__name__}actor/")
            self.actor.summary(print_fn=self.logger.info)

    def build_actor(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        observ = Input(shape=(observDim,), dtype=self.tfDtype, name="actor_inputs")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"actor_hidden_{ix}")(h)
        action = self.dense_or_batchNorm(actionDim, "tanh", use_bias=False, trainable=trainable, name="actor_outputs")(h)

        net = Model(inputs=observ, outputs=action, name="actor")
        return net

    def build_critic(self, observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, dtype, trainable=True):
        observ_inputs = Input(shape=(observDim,), dtype=dtype, name="critic_observ_inputs")
        h = observ_inputs
        for ix, units in enumerate(observ_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"critic_observ_hidden_{ix}")(h)
        observ_outputs = h

        action_inputs = Input(shape=(actionDim,), dtype=dtype, name="critic_action_inputs")
        h = action_inputs
        for ix, units in enumerate(action_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"critic_action_hidden_{ix}")(h)
        action_outputs = h

        concat_inputs = Concatenate(trainable=trainable)([observ_outputs, action_outputs]) 

        h = concat_inputs
        for ix, units in enumerate(concat_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"critic_concat_hidden_{ix}")(h)
        Q = self.dense_or_batchNorm(1, "linear", trainable=trainable, name="critic_outputs")(h)

        net = Model(inputs=[observ_inputs, action_inputs], outputs=Q, name="critic")
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
            action = self.actor(observ)             # shape=(batchSz,actionDim)
            Q1 = self.critic1([observ, action])     # shape=(batchSz,1)
            actor_loss = -tf.reduce_mean(Q1)        # shape=()
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def update_actor_with2critics(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            action = self.actor(observ)             # shape=(batchSz,actionDim)
            Q1 = self.critic1([observ, action])     # shape=(batchSz,1)
            Q2 = self.critic2([observ, action])     # shape=(batchSz,1)
            Q_min = tf.minimum(Q1, Q2)              # shape=(batchSz,1)
            actor_loss = -tf.reduce_mean(Q_min)     # shape=()
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def update_critic(self, observ, action, reward, next_observ, done, importance_weights):
        """ Args:
            observ, next_observ: shape=(batchSz,observDim)
            action: shape=(batchSz,actionDim)
            done, reward: shape=(batchSz,1)
        """
        with tf.GradientTape() as tape:
            next_action = self.target_actor(next_observ) if self.isTargetActor else self.actor(next_observ)  # shape=(batchSz,actionDim)
            target_Q = self.target_critic1([next_observ, next_action])  # shape=(batchSz,1)
            y = reward + (1.0 - done) * self.gamma * target_Q  # shape=(batchSz,1)
            Q = self.critic1([observ, action])  # shape=(batchSz,1)
            td_error = tf.square(y - Q)  # shape=(batchSz,1)
            if self.isPER:
                critic1_loss = tf.reduce_mean(importance_weights * td_error)  # shape=()
            else:
                critic1_loss = tf.reduce_mean(td_error)  # shape=()
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        return critic1_loss, td_error

    @tf.function
    def update_2critics(self, observ, action, reward, next_observ, done, importance_weights):
        """
        Args:
            observ, next_observ: shape=(batchSz,observDim)
            action: shape=(batchSz,actionDim)
            done, reward: shape=(batchSz,1)
        """
        with tf.GradientTape(persistent=True) as tape:
            next_action = self.target_actor(next_observ) if self.isTargetActor else self.actor(next_observ)  # shape=(batchSz,actionDim)

            target_Q1 = self.target_critic1([next_observ, next_action])  # shape=(batchSz,1)
            target_Q2 = self.target_critic2([next_observ, next_action])  # shape=(batchSz,1)
            target_Q_min = tf.minimum(target_Q1, target_Q2)  # shape=(batchSz,1)
            y = reward + (1.0 - done) * self.gamma * target_Q_min  # shape=(batchSz,1)

            Q1 = self.critic1([observ, action])  # shape=(batchSz,1)
            Q2 = self.critic2([observ, action])  # shape=(batchSz,1)
            td_error1 = tf.square(y - Q1)  # shape=(batchSz,1)
            td_error2 = tf.square(y - Q2)  # shape=(batchSz,1)
            td_error = tf.minimum(td_error1, td_error2)  # shape=(batchSz,1)
            if self.isPER:
                critic1_loss = tf.reduce_mean(importance_weights * td_error1)  # shape=()
                critic2_loss = tf.reduce_mean(importance_weights * td_error2)  # shape=()
            else:
                critic1_loss = tf.reduce_mean(td_error1)  # shape=()
                critic2_loss = tf.reduce_mean(td_error2)  # shape=()
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        return critic1_loss, critic2_loss, td_error

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
        if self.isCritic2:
            critic1_loss, critic2_loss, td_error = self.update_2critics(
                    batch.observ,
                    batch.action,
                    batch.reward,
                    batch.next_observ,
                    batch.done,
                    importance_weights
            )
            actor_loss = self.update_actor_with2critics(batch.observ)
        else:
            critic1_loss, td_error = self.update_critic(
                    batch.observ,
                    batch.action,
                    batch.reward,
                    batch.next_observ,
                    batch.done,
                    importance_weights
            )
            actor_loss = self.update_actor(batch.observ)

        self.soft_update()
        return (critic1_loss, actor_loss), td_error  # (,) to be consistent with DQN return

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: 1d ndarray
        return:
            action: 1d ndarray
        """
        if (self.mode == "train") and not self.isReadyToTrain():
            actionToEnv = actionCoder.random_vec()
            action = actionCoder.encode(actionToEnv)
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)  # shape=(1,observDim) to input to net

            action = self.actor(observ)[0]  # actor() returns [action]
        return action

    def isReadyToTrain(self):
        b1 = self.mode == "train"
        b2 = self.replayBuffer.memoryCnt > self.memoryCnt_toStartTrain
        b3 = self.replayBuffer.memoryCnt > self.batchSz
        #   b4 = self.actCnt % 1 == 0
        return b1 and b2 and b3  #   and b4

    def save(self, msg=""):
        self.actor.save(f"{self.savePath}/{self.__class__.__name__}/actor/")
        self.critic1.save(f"{self.savePath}/{self.__class__.__name__}/critic1/")
        self.target_critic1.save(f"{self.savePath}/{self.__class__.__name__}/target_critic1/")
        if self.isTargetActor:
            self.target_actor.save(f"{self.savePath}/{self.__class__.__name__}/target_actor/")
        if self.isCritic2:
            self.critic2.save(f"{self.savePath}/{self.__class__.__name__}/critic2/")
            self.target_critic2.save(f"{self.savePath}/{self.__class__.__name__}/target_critic2/")
        self.replayBuffer.save(f"{self.savePath}/{self.__class__.__name__}/replayBuffer.json")
        self.logger.info(msg)

    def summary(self):
        self.actor.summary(print_fn=self.logger.info)  # to print in logger file
        self.critic1.summary(print_fn=self.logger.info)  # to print in logger file
        
    def summaryWrite(self, key, value, step):
        with self.writer.as_default():
            tf.summary.scalar(key, value, step=step)

