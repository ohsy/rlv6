"""
2023.05.02 made by soh

- Made from Agent_ddpg.py and github BoltzmannDRL
- Agent_sac is changed into SAC, which is used in game.py
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
from importlib import import_module
from DDPG import DDPG

class SAC(DDPG):
    def __init__(self, envName, mode, config, logger, observDim, actionDim):
        super().__init__(envName, mode, config, logger, observDim, actionDim)
        self.tiny = 1e-6  # added to denominator to prevent inf; NOTE: value < 1e-6 (like 1e-7) is considered as 0 
        self.logStd_min = -13  # e**(-13) = 2.26e-06; for stds
        self.logStd_max = 1

    def build_actor(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        observ = Input(shape=(observDim,), dtype=dtype, name="observ")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)
        mean = self.dense_or_batchNorm(actionDim, "tanh", use_bias=False, trainable=trainable, name="mean")(h)
        logStd = self.dense_or_batchNorm(actionDim, "linear", trainable=trainable, name="logStd")(h)

        net = Model(inputs=observ, outputs=[mean, logStd], name="actor")
        net.compile(optimizer=Adam(learning_rate=self.actor_lr)) # wo this save() saves one outputs; not mean & logStd
        #   net.compile()  # NOTE: without this save() saves one outputs instead of mean and logStd; not working??
        return net

    def get_action_logProb(self, observ, withTarget=False):
        mean, logStd = self.target_actor(observ) if withTarget else self.actor(observ)  # each (batchSz,actionDim)
        logStd = tf.clip_by_value(logStd, self.logStd_min, self.logStd_max)             # (batchSz,actionDim)
        std = tf.exp(logStd)                                                            # (batchSz,actionDim)

        dist = tfp.distributions.Normal(mean, std)                                      # batchSz distributions
        actionSampled = dist.sample()                                                   # (batchSz,actionDim)
        action = tf.tanh(actionSampled) # squashing to be in (-1,1)                     # (batchSz,actionDim)

        logProb_of_actionSampled = dist.log_prob(actionSampled)                         # (batchSz,actionDim)
        logProb = logProb_of_actionSampled - tf.math.log(1 - action**2 + self.tiny)     # logProb of action. 
                                                                                        # cf. eq.(21) of 2018 SAC paper
        logProb = tf.reduce_sum(logProb, axis=1, keepdims=True)  # sum over multiple action parameters; (batchSz,1)

        return action, logProb

    @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            action, logProb = self.get_action_logProb(observ)   # each (batchSz,1)
            Q1 = self.critic1([observ, action])                 # (batchSz,1)
            if self.isCritic2:
                Q2 = self.critic2([observ, action])             # (batchSz,1)
                Q_min = tf.minimum(Q1, Q2)                      # (batchSz,1)
                Q_soft = Q_min - self.alpha * logProb           # (batchSz,1)
            else:
                Q_soft = Q1 - self.alpha * logProb              # (batchSz,1)
            actor_loss = -tf.reduce_mean(Q_soft)                # ()

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
        with tf.GradientTape(persistent=True) as tape:
            next_action, next_logProb = self.get_action_logProb(next_observ, withTarget=self.isTargetActor) # (batchSz,1)
            target_Q1 = self.target_critic1([next_observ, next_action])     # (batchSz,1)
            if self.isCritic2:
                target_Q2 = self.target_critic2([next_observ, next_action]) # (batchSz,1)
                target_Q_min = tf.minimum(target_Q1, target_Q2)             # (batchSz,1)
                target_Q_soft = target_Q_min - self.alpha * next_logProb    # (batchSz,1)
                y = reward + (1.0 - done) * self.gamma * target_Q_soft      # (batchSz,1)

                Q2 = self.critic2([observ, action])                         # (batchSz,1)
                td_error2 = tf.square(y - Q2)                               # (batchSz,1)
                td_error2 = importance_weights * td_error2 if self.isPER else td_error2
                critic2_loss = tf.reduce_mean(td_error2)                    # ()
            else:
                target_Q_soft = target_Q1 - self.alpha * next_logProb       # (batchSz,1)
                y = reward + (1.0 - done) * self.gamma * target_Q_soft      # (batchSz,1)
            Q1 = self.critic1([observ, action])                             # (batchSz,1)
            td_error1 = tf.square(y - Q1)                                   # (batchSz,1)
            td_error1 = importance_weights * td_error1 if self.isPER else td_error1
            critic1_loss = tf.reduce_mean(td_error1)                        # ()

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        if self.isCritic2:
            critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
                #   td_error = tf.minimum(td_error1, td_error2)                     # for monitoring; (batchSz,1)
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
            observ = tf.expand_dims(observ, axis=0)         # (1,observDim) to input to net
            action, _ = self.get_action_logProb(observ)     # (batchSz,actionDim)
            action = action[0]                              # (actionDim)
        return action

