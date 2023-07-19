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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from replaybuffer import ReplayBuffer, PERBuffer
from importlib import import_module
from DDPG import DDPG


class SAC_entropy(DDPG):
    def __init__(self, envName, mode, config, logger, observ_nNodes, action_nNodes):
        super().init_parameters(envName, mode, config, logger)
        
        observDim = sum(observ_nNodes)
        actionDim = sum(action_nNodes)

        if mode == "train":
            self.actor = self.build_actor(observDim, self.actor_hiddenUnits, action_nNodes, self.tfDtype)
            self.critic1 = self.build_critic(
                    observDim, self.observ_hiddenUnits, actionDim, self.action_hiddenUnits, self.concat_hiddenUnits, self.tfDtype)
            self.target_critic1 = self.build_critic(
                    observDim, self.observ_hiddenUnits, actionDim, self.action_hiddenUnits, self.concat_hiddenUnits, self.tfDtype,
                    trainable=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            if self.isTargetActor:
                self.target_actor = self.build_actor(observDim, actor_hiddenUnits, action_nNodes, self.tfDtype,
                    trainable=False)
            if self.isCritic2:
                self.critic2 = self.build_critic(
                        observDim, self.observ_hiddenUnits, actionDim, self.action_hiddenUnits, self.concat_hiddenUnits, self.tfDtype)
                self.target_critic2 = self.build_critic(
                        observDim, self.observ_hiddenUnits, actionDim, self.action_hiddenUnits, self.concat_hiddenUnits, self.tfDtype,
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
        """ softmax activation, Softmax layer results in NaN. So I use my own. """
        observ = Input(shape=(observDim,), dtype=dtype, name="observ")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)

        probs = []
        for ix in range(len(action_nNodes)):
            logit = self.dense_or_batchNorm(action_nNodes[ix], "linear", trainable=trainable, name=f"logit{ix}")(h)
            logit = tf.clip_by_value(logit, self.logit_min, self.logit_max)  # before exp() to prevent gradient explosion
            prob = Softmax()(logit)  
                #   exps = tf.math.exp(logit)
                #   sums = tf.reduce_sum(exps, axis=1, keepdims=True) + self.tiny                    # tiny to prevent NaN
                #   prob = exps / sums    # softmax
            probs.append(prob)
        action_asProb = Concatenate(trainable=trainable, name="action_asProb")(probs) if len(action_nNodes) > 1 else probs[0]

        net = Model(inputs=observ, outputs=action_asProb, name="actor")
            #   net.compile(optimizer=Adam(learning_rate=self.actor_lr)) # wo this save() saves one outputs; not mean & logStd
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

    """
    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
       
        Args:
            observ: shape=(observDim)
        return:
            action: shape=(actionDim)
      
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
    """
