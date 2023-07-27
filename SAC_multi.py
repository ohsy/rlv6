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
from replaymemory import ReplayMemory, PERMemory
from importlib import import_module
from DDPG import DDPG


class SAC_multi(DDPG):
    def __init__(self, envName, mode, config, logger, observ_nNodes, action_nNodes):
        self.observ_nNodes = observ_nNodes
        self.action_nNodes = action_nNodes        
        observDim = sum(observ_nNodes)
        actionDim = sum(action_nNodes)

        super().__init__(envName, mode, config, logger, observDim, actionDim)

    def build_actor(self, trainable=True):  # , observDim, hiddenUnits, action_nNodes, dtype, trainable=True):
        """ softmax activation, Softmax layer results in NaN. So I use my own. """
        observ = Input(shape=(self.observDim,), dtype=self.tfDtype, name="observ")
        h = observ
        for ix, units in enumerate(self.actor_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)

        probs = []
        for ix in range(len(self.action_nNodes)):
            logit = self.dense_or_batchNorm(self.action_nNodes[ix], "linear", trainable=trainable, name=f"logit{ix}")(h)
            logit = tf.clip_by_value(logit, self.logit_min, self.logit_max)  # before exp() to prevent gradient explosion
            prob = Softmax()(logit)  
                #   exps = tf.math.exp(logit)
                #   sums = tf.reduce_sum(exps, axis=1, keepdims=True) + self.tiny                    # tiny to prevent NaN
                #   prob = exps / sums    # softmax
            probs.append(prob)
        action_asProb = Concatenate(trainable=trainable, name="action_asProb")(probs) if len(self.action_nNodes) > 1 else probs[0]

        net = Model(inputs=observ, outputs=action_asProb, name="actor")
            #   net.compile(optimizer=Adam(learning_rate=self.actor_lr)) # wo this save() saves one outputs; not mean & logStd
        return net

    def get_action_asProb_entropy(self, observ):
        action_asProb = self.actor(observ)    # softmax; (batchSz,actionDim)
        self.logger.debug(f"in get_action_asProb_entropy: action_asProb={action_asProb}")
        logProb = tf.math.log(action_asProb)                                               # (batchSz,actionDim)
        entropy = tf.reduce_sum(-action_asProb * logProb, axis=1, keepdims=True)           # (batchSz,1)

        return action_asProb, entropy

    # @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            action_asProb, entropy = self.get_action_asProb_entropy(observ)   # (batchSz,actionDim), (batchSz,1)
            self.logger.debug(f"action_asProb={action_asProb}")
            self.logger.debug(f"entropy={entropy}")
            Q1 = self.critic1([observ, action_asProb])              # (batchSz,1)
            Q2 = self.critic2([observ, action_asProb])              # (batchSz,1)
            self.logger.debug(f"Q1={Q1}")
            self.logger.debug(f"Q2={Q2}")
            Q_min = tf.minimum(Q1, Q2)                              # (batchSz,1)
            self.logger.debug(f"Q_min={Q_min}")
            Q_soft = Q_min + self.alpha * entropy                   # (batchSz,1); NOTE: alpha term is added, not subtracted
            self.logger.debug(f"Q_soft={Q_soft}")
            actor_loss = -tf.reduce_mean(Q_soft)                        # ()
            self.logger.debug(f"actor_loss={actor_loss}")

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    # @tf.function
    def update_critic(self, observ, action, reward, next_observ, done, importance_weights):
        """ Args:
            observ, next_observ: shape=(batchSz,observDim) 
            action: shape=(batchSz,actionDim)
            done, reward: shape=(batchSz,1)
        """
        with tf.GradientTape(persistent=True) as tape:
            next_action, next_entropy = self.get_action_asProb_entropy(next_observ) # (batchSz,actionDim), (batchSz,1)
            self.logger.debug(f"next_action={next_action}")
            self.logger.debug(f"next_entropy={next_entropy}")
            target_Q1 = self.target_critic1([next_observ, next_action]) # (batchSz,1)
            target_Q2 = self.target_critic2([next_observ, next_action]) # (batchSz,1)
            self.logger.debug(f"target_Q1={target_Q1}")
            self.logger.debug(f"target_Q2={target_Q2}")
            target_Q_min = tf.minimum(target_Q1, target_Q2)             # (batchSz,1)
            self.logger.debug(f"target_Q_min={target_Q_min}")
            target_Q_soft = target_Q_min + self.alpha * next_entropy    # (batchSz,1)
            self.logger.debug(f"target_Q_soft={target_Q_soft}")
            y = reward + (1.0 - done) * self.gamma * target_Q_soft      # (batchSz,1)
            self.logger.debug(f"y={y}")

            Q1 = self.critic1([observ, action])                         # (batchSz,1)
            self.logger.debug(f"Q1={Q1}")
            td_error1 = tf.square(y - Q1)                               # (batchSz,1)
            self.logger.debug(f"td_error1={td_error1}")
            td_error1 = importance_weights * td_error1 if self.isPER else td_error1
            self.logger.debug(f"td_error1={td_error1}")
            critic1_loss = tf.reduce_mean(td_error1)                    # ()
            self.logger.debug(f"critic1_loss={critic1_loss}")

            Q2 = self.critic2([observ, action])                         # (batchSz,1)
            self.logger.debug(f"Q2={Q2}")
            td_error2 = tf.square(y - Q2)                               # (batchSz,1)
            self.logger.debug(f"td_error2={td_error2}")
            td_error2 = importance_weights * td_error2 if self.isPER else td_error2
            self.logger.debug(f"td_error2={td_error2}")
            critic2_loss = tf.reduce_mean(td_error2)                    # ()
            self.logger.debug(f"critic2_loss={critic2_loss}")

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))

        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
            #   td_error = tf.minimum(td_error1, td_error2)                     # for monitoring; (batchSz,1)
        return critic1_loss, critic2_loss, td_error1

