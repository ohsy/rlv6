"""
2023.05.14 made by soh 

- from SAC.py and SAC_discrete.py
"""
import sys
import json
#   from tqdm import tqdm
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import TensorBoard
from replaybuffer import ReplayBuffer, PERBuffer
from importlib import import_module
from Agent import Agent


class SAC_entropy(Agent):
    def __init__(self, envName, mode, config, logger, observ_nNodes, action_nNodes):
        super().__init__(envName, mode, config, logger)
        
        observDim = sum(observ_nNodes)
        actionDim = sum(action_nNodes)
        self.actor_lr = config["Actor_learningRate"]
        self.critic_lr = config["Critic_learningRate"]

        actor_hiddenUnits = config["Actor_hiddenUnits"]                     # like [64, 'bn', 64], 'bn' for BatchNorm
        observ_hiddenUnits = config["Critic_observationBlock_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm
        action_hiddenUnits = config["Critic_actionBlock_hiddenUnits"]       # like [64, 'bn', 64], 'bn' for BatchNorm
        concat_hiddenUnits = config["Critic_concatenateBlock_hiddenUnits"]  # like [64, 'bn', 64], 'bn' for BatchNorm

        if mode == "train":
            self.actor = self.build_actor(observDim, actor_hiddenUnits, action_nNodes, self.tfDtype)
            self.critic1 = self.build_critic(
                    observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype)
            self.target_critic1 = self.build_critic(
                    observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype,
                    trainable=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = self.build_actor(observDim, actor_hiddenUnits, action_nNodes, self.tfDtype,
                    trainable=False)
            if self.isCritic2:
                self.critic2 = self.build_critic(
                        observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype)
                self.target_critic2 = self.build_critic(
                        observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits, self.tfDtype,
                        trainable=False)
                self.critic2_optimizer = Adam(self.critic_lr)
        elif mode == "test":
            self.actor = load_model(f"{self.savePath}/actor/", compile=False)
            self.actor.summary(print_fn=self.logger.info)
        elif mode == "continued_train":
            self.actor = load_model(f"{self.savePath}/actor/", compile=False)
            self.critic1 = load_model(f"{self.savePath}/critic1/", compile=False)
            self.target_critic1 = load_model(f"{self.savePath}/target_critic1/", compile=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isCritic2:
                self.critic2 = load_model(f"{self.savePath}/critic2/", compile=False)
                self.target_critic2 = load_model(f"{self.savePath}/target_critic2/", compile=False)
                self.critic2_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = load_model(f"{self.savePath}/target_actor/", compile=False)
            self.actor.summary(print_fn=self.logger.info)
            self.critic1.summary(print_fn=self.logger.info)
            self.explorer.load()


    def build_actor(self, observDim, hiddenUnits, action_nNodes, dtype, trainable=True):
        """ softmax activation, Softmax layer results in NaN """
        observ = Input(shape=(observDim,), dtype=dtype, name="observ")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)

        probs = []
        for ix in range(len(action_nNodes)):
            logit = self.dense_or_batchNorm(action_nNodes[ix], "linear", use_bias=True, trainable=trainable, name="logit")(h)
                #   action_asProb = Softmax()(logit)  # causes problem sometimes
            exps = tf.math.exp(logit)
            sums = tf.reduce_sum(exps, axis=1, keepdims=True) + self.tiny                    # tiny to prevent NaN
            prob = exps / sums    # softmax
            probs.append(prob)
        action_asProb = Concatenate(trainable=trainable, name="action_asProb")(probs)        

        net = Model(inputs=observ, outputs=action_asProb, name="actor")
        return net

    def build_critic(self, observDim, observ_hiddenUnits, actionDim, action_hiddenUnits, concat_hiddenUnits,
                     dtype, trainable=True):
        observ_inputs = Input(shape=(observDim,), dtype=dtype, name="observ_in")
        h = observ_inputs
        for ix, units in enumerate(observ_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"observ_hidden_{ix}")(h)
        observ_outputs = h

        action_inputs = Input(shape=(actionDim,), dtype=dtype, name="action_in")
        h = action_inputs
        for ix, units in enumerate(action_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"action_hidden_{ix}")(h)
        action_outputs = h

        concat_inputs = Concatenate(trainable=trainable)([observ_outputs, action_outputs])

        h = concat_inputs
        for ix, units in enumerate(concat_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"concat_hidden_{ix}")(h)
        Q = self.dense_or_batchNorm(1, "linear", trainable=trainable, name="out")(h)

        net = Model(inputs=[observ_inputs, action_inputs], outputs=Q, name="critic")
        return net

    def get_action_asProb_entropy(self, observ, withTarget=False):
        action_asProb = self.target_actor(observ) if withTarget else self.actor(observ)    # softmax; (batchSz,actionDim)
        self.logger.debug(f"in get_action_asProb_entropy: action_asProb={action_asProb}")
        logProb = tf.math.log(action_asProb)                                               # (batchSz,actionDim)
        entropy = tf.reduce_sum(-action_asProb * logProb, axis=1, keepdims=True)           # (batchSz,1)

        return action_asProb, entropy

    """
    def get_action(self, observ, isStochastic=False):
        Args:
            observ: shape=(observDim)

        action_asProb = self.actor(observ)  # softmax; (batchSz,actionDim)
        self.logger.debug(f"in get_action: action_asProb={action_asProb}")
        if isStochastic: 
            dist = tfp.distributions.Multinomial(total_count=1, probs=action_asProb)         # batchSz distributions
            action = dist.sample()              # one-hot vector; (batchSz,actionDim)
        else:
            maxIdx = tf.argmax(action_asProb, axis=1)    # (batchSz)
            action = tf.one_hot(maxIdx, self.actionDim, dtype=self.tfDtype)  # (batchSz,actionDim)
                #   maxProb = tf.reduce_max(action_asProb, axis=1, keepdims=True)    # (batchSz,1)
        return action
    """

    @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            action_asProb, entropy = self.get_action_asProb_entropy(observ)   # (batchSz,actionDim), (batchSz,1)
            Q1 = self.critic1([observ, action_asProb])                     # (batchSz,1)
            if self.isCritic2:
                Q2 = self.critic2([observ, action_asProb])                 # (batchSz,1)
                Q_min = tf.minimum(Q1, Q2)                              # (batchSz,1)
                Q_soft = Q_min + self.alpha * entropy                   # (batchSz,1); NOTE: alpha term is added, not subtracted
            else:
                Q_soft = Q1 + self.alpha * entropy                      # (batchSz,1); NOTE: alpha term is added, not subtracted
            actor_loss = -tf.reduce_mean(Q_soft)                        # ()

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def update_critic(self, observ, action, reward, next_observ, done, importance_weights):
        """ Args:
            observ, next_observ: shape=(batchSz,observDim) 
            action: one-hot vector for actionToEnv in possibleValuesFor[]; shape=(batchSz,actionDim) 
            done, reward: shape=(batchSz,1)
        """
        with tf.GradientTape(persistent=True) as tape:
            next_action_asProb, next_entropy = self.get_action_asProb_entropy(next_observ, withTarget=self.isTargetActor) 
                    # (batchSz,actionDim), (batchSz,1)
            target_Q1 = self.target_critic1([next_observ, next_action_asProb])     # (batchSz,1)
            if self.isCritic2:
                target_Q2 = self.target_critic2([next_observ, next_action_asProb]) # (batchSz,1)
                target_Q_min = tf.minimum(target_Q1, target_Q2)                 # (batchSz,1)
                target_Q_soft = target_Q_min + self.alpha * next_entropy        # (batchSz,1); NOTE: alpha term is added, not subtracted
                y = reward + (1.0 - done) * self.gamma * target_Q_soft          # (batchSz,1)

                Q2 = self.critic2([observ, action])
                td_error2 = tf.square(y - Q2)             
                td_error2 = importance_weights * td_error2 if self.isPER else td_error2
                critic2_loss = tf.reduce_mean(td_error2)
            else:
                target_Q_soft = target_Q1 + self.alpha * next_entropy           # (batchSz,1); NOTE: alpha term is added, not subtracted
                y = reward + (1.0 - done) * self.gamma * target_Q_soft          # (batchSz,1)

            Q1 = self.critic1([observ, action])                                 # (batchSz,1)
            td_error1 = tf.square(y - Q1)                                       # (batchSz,1)
            td_error1 = importance_weights * td_error1 if self.isPER else td_error1
            critic1_loss = tf.reduce_mean(td_error1)                            # ()

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        if self.isCritic2:
            critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
                #   td_error = tf.minimum(td_error1, td_error2)                     # (batchSz,1)
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
            action: shape=(actionDim)
        """
        if self.explorer.isReadyToExplore():
                #   actionToEnv = actionCoder.random_decoded()
                #   action = actionCoder.encode(actionToEnv)
            action = actionCoder.random_encoded()
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)                     # (1,observDim) to input to net
                #   action = self.get_action(observ, self.isActionStochastic)   # (1,actionDim) 
            action_asProb = self.actor(observ)                          # softmax; (batchSz,actionDim)
            action = action_asProb[0]                                          # (actionDim)
        return action

        
    def save(self):
        self.actor.save(f"{self.savePath}/actor/")
        self.critic1.save(f"{self.savePath}/critic1/")
        self.target_critic1.save(f"{self.savePath}/target_critic1/")
        if self.isTargetActor:
            self.target_actor.save(f"{self.savePath}/target_actor/")
        if self.isCritic2:
            self.critic2.save(f"{self.savePath}/critic2/")
            self.target_critic2.save(f"{self.savePath}/target_critic2/")
        self.replayBuffer.save()
        self.explorer.save()

    def summary(self):
        self.actor.summary(print_fn=self.logger.info)   # to print in logger file
        self.critic1.summary(print_fn=self.logger.info) # to print in logger file


