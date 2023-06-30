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
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from replaybuffer import ReplayBuffer, PERBuffer
from importlib import import_module
from DDPG import DDPG


class SAC_entropy(DDPG):
    def __init__(self, envName, mode, config, logger, observDim, actionDim):
        super().__init__(envName, mode, config, logger, observDim, actionDim)
        self.actionDim = actionDim

    def build_actor(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        """ softmax activation, Softmax layer results in NaN """
        observ = Input(shape=(observDim,), dtype=dtype, name="observ")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)
        logit = self.dense_or_batchNorm(actionDim, "linear", use_bias=True, trainable=trainable, name="logit")(h)
            #   actionProb = Softmax()(logit)
        exps = tf.math.exp(logit)
        sums = tf.reduce_sum(exps, axis=1, keepdims=True) + self.tiny                    # eps to prevent NaN
        actionProb = exps / sums    # softmax

        net = Model(inputs=observ, outputs=actionProb, name="actor")
        return net

    def get_actionProb_entropy(self, observ, withTarget=False):
        actionProb = self.target_actor(observ) if withTarget else self.actor(observ)    # softmax; (batchSz,actionDim)
        self.logger.debug(f"in get_actionProb_entropy: actionProb={actionProb}")
        logProb = tf.math.log(actionProb)                                               # (batchSz,actionDim)
        entropy = tf.reduce_sum(-actionProb * logProb, axis=1, keepdims=True)           # (batchSz,1)

        return actionProb, entropy

    def get_action(self, observ, isStochastic=False):
        """
        Args:
            observ: shape=(observDim)
        """
        actionProb = self.actor(observ)  # softmax; (batchSz,actionDim)
        self.logger.debug(f"in get_action: actionProb={actionProb}")
        if isStochastic: 
            dist = tfp.distributions.Multinomial(total_count=1, probs=actionProb)         # batchSz distributions
            action = dist.sample()              # one-hot vector; (batchSz,actionDim)
        else:
            maxIdx = tf.argmax(actionProb, axis=1)    # (batchSz)
            action = tf.one_hot(maxIdx, self.actionDim, dtype=self.tfDtype)  # (batchSz,actionDim)
                #   maxProb = tf.reduce_max(actionProb, axis=1, keepdims=True)    # (batchSz,1)
        return action

    @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            actionProb, entropy = self.get_actionProb_entropy(observ)   # (batchSz,actionDim), (batchSz,1)
            Q1 = self.critic1([observ, actionProb])                     # (batchSz,1)
            if self.isCritic2:
                Q2 = self.critic2([observ, actionProb])                 # (batchSz,1)
                Q_min = tf.minimum(Q1, Q2)                              # (batchSz,1)
                Q_soft = Q_min + self.alpha * entropy                   # (batchSz,1)
            else:
                Q_soft = Q1 + self.alpha * entropy                      # (batchSz,1)
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
            next_actionProb, next_entropy = self.get_actionProb_entropy(next_observ, withTarget=self.isTargetActor) 
                    # (batchSz,actionDim), (batchSz,1)
            target_Q1 = self.target_critic1([next_observ, next_actionProb])     # (batchSz,1)
            if self.isCritic2:
                target_Q2 = self.target_critic2([next_observ, next_actionProb]) # (batchSz,1)
                target_Q_min = tf.minimum(target_Q1, target_Q2)                 # (batchSz,1)
                target_Q_soft = target_Q_min + self.alpha * next_entropy        # (batchSz,1)
                y = reward + (1.0 - done) * self.gamma * target_Q_soft          # (batchSz,1)

                Q2 = self.critic2([observ, action])
                td_error2 = tf.square(y - Q2)             
                td_error2 = importance_weights * td_error2 if self.isPER else td_error2
                critic2_loss = tf.reduce_mean(td_error2)
            else:
                target_Q_soft = target_Q1 + self.alpha * next_entropy           # (batchSz,1)
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

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: shape=(observDim)
        return:
            action: shape=(actionDim)
        """
        if self.explorer.isReadyToExplore():
            actionToEnv = actionCoder.random_decoded()
            action = actionCoder.encode(actionToEnv)
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)                     # (1,observDim) to input to net
            action = self.get_action(observ, self.isActionStochastic)   # (1,actionDim) 
            action = action[0]                                          # (actionDim)
        return action

