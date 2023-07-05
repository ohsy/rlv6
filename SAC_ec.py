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
import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import TensorBoard
from replaybuffer import ReplayBuffer, PERBuffer
from importlib import import_module
from DDPG import DDPG

class SAC_ec(DDPG):
    def __init__(self, envName, mode, config, logger, observDim, actionSz):
        """
        Args: actionSz: num of action parameters
        """
        actionDim = 2 * actionSz
        super().__init__(envName, mode, config, logger, observDim, actionDim)
        self.logger.debug(f"actionSz={actionSz}, actionDim={actionDim}")
        self.actionSz = actionSz
        self.tiny = 1e-6  # added to denominator to prevent inf; NOTE: value < 1e-6 (like 1e-7) is considered as 0 
        self.logStd_min = -13  # e**(-13) = 2.26e-06; for stds
        self.logStd_max = 1
        self.log_2pie_over2 = tf.constant(math.log(2 * math.pi * math.e) / 2) 

    def build_actor(self, observDim, hiddenUnits, actionDim, dtype, trainable=True):
        actionSz = int(actionDim/2)
        observ = Input(shape=(observDim,), dtype=dtype, name="observ")
        h = observ
        for ix, units in enumerate(hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)
        mean = self.dense_or_batchNorm(actionSz, "linear", use_bias=False, trainable=trainable, name="mean")(h)
        logStd = self.dense_or_batchNorm(actionSz, "linear", trainable=trainable, name="logStd")(h)

        net = Model(inputs=observ, outputs=[mean, logStd], name="actor")
        net.compile(optimizer=Adam(learning_rate=self.actor_lr)) # wo this save() saves one outputs; not mean & logStd

        return net

    def get_action_entropy(self, observ, withTarget=False):
        mean, logStd = self.target_actor(observ) if withTarget else self.actor(observ)  # each (batchSz,actionSz)
        logStd = tf.clip_by_value(logStd, self.logStd_min, self.logStd_max)             # (batchSz,actionSz)

        entropy = self.log_2pie_over2 + logStd

        action = tf.concat([mean, logStd], axis=1)
        self.logger.debug(f"mean={mean}, logStd={logStd}, action={action}")

        return action, entropy

    @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            action, entropy = self.get_action_entropy(observ)   # each (batchSz,1)
            Q1 = self.critic1([observ, action])                 # (batchSz,1)
            if self.isCritic2:
                Q2 = self.critic2([observ, action])             # (batchSz,1)
                Q_min = tf.minimum(Q1, Q2)                      # (batchSz,1)
                Q_soft = Q_min - self.alpha * entropy           # (batchSz,1)
            else:
                Q_soft = Q1 - self.alpha * entropy              # (batchSz,1)
            actor_loss = -tf.reduce_mean(Q_soft)                # ()

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def update_critic(self, observ, action_mean, reward, next_observ, done, importance_weights):
        """ Args:
            observ, next_observ: shape=(batchSz,observDim) 
            action_mean: shape=(batchSz,actionSz)
            done, reward: shape=(batchSz,1)
        """
        action_logStd = self.tiny * tf.ones_like(action_mean)
        action = tf.concat([action_mean, action_logStd], axis=1)	# (batchSz,actionDim)
        with tf.GradientTape(persistent=True) as tape:
            next_action, next_entropy = self.get_action_entropy(next_observ, withTarget=self.isTargetActor) # (batchSz,1)
            self.logger.debug(f"next_observ={next_observ}, next_action={next_action}")
            target_Q1 = self.target_critic1([next_observ, next_action])     # (batchSz,1)


            if self.isCritic2:
                target_Q2 = self.target_critic2([next_observ, next_action]) # (batchSz,1)
                target_Q_min = tf.minimum(target_Q1, target_Q2)             # (batchSz,1)
                target_Q_soft = target_Q_min - self.alpha * next_entropy    # (batchSz,1)
                y = reward + (1.0 - done) * self.gamma * target_Q_soft      # (batchSz,1)

                Q2 = self.critic2([observ, action])                         # (batchSz,1)
                td_error2 = tf.square(y - Q2)                               # (batchSz,1)
                td_error2 = importance_weights * td_error2 if self.isPER else td_error2
                critic2_loss = tf.reduce_mean(td_error2)                    # ()
            else:
                target_Q_soft = target_Q1 - self.alpha * next_entropy       # (batchSz,1)
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
            action_mean = actionCoder.encode(actionToEnv)
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)         # (1,observDim) to input to net
            action, _ = self.get_action_entropy(observ)     # (batchSz,actionDim)
            action_mean = action[0][:self.actionSz]         # (actionSz)
        return action_mean

