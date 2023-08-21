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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from replaymemory import ReplayMemory, PERMemory
from importlib import import_module
from ActorCritic import ActorCritic


class SAC_discrete(ActorCritic):
    def __init__(self, envName, mode, config, logger, observDim, actionDim):
        self.actor_hiddenUnits = config["Actor_hiddenUnits"]     # like [64, 'bn', 64], 'bn' for BatchNorm
        self.critic_hiddenUnits = config["Critic_hiddenUnits"]   # like [64, 'bn', 64], 'bn' for BatchNorm
        super().__init__(envName, mode, config, logger, observDim, actionDim)

    def build_actor(self, trainable=True):
        observ = Input(shape=(self.observDim,), dtype=self.tfDtype, name="observ")
        h = observ
        for ix, units in enumerate(self.actor_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)
        #   actionProb = self.dense_or_batchNorm(self.actionDim, "softmax", trainable=trainable, name="actionProb")(h)
        logit = self.dense_or_batchNorm(self.actionDim, "linear", trainable=trainable, name="logit")(h)
        logit = tf.clip_by_value(logit, self.logit_min, self.logit_max)  # before exp() to prevent gradient explosion
        actionProb = Softmax()(logit)
            #   exps = tf.math.exp(logit)
            #   sums = tf.reduce_sum(exps, axis=1, keepdims=True) + self.tiny                    # tiny to prevent NaN
            #   actionProb = exps / sums    # softmax

        net = Model(inputs=observ, outputs=actionProb, name="actor")
        return net

    def get_actionProb_logActionProb(self, observ, withTarget=False):
        prob = self.target_actor(observ) if withTarget else self.actor(observ)  # (batchSz,actionDim)
        self.logger.debug(f"in get_actionProb_logActionProb:prob={prob}")
        logProb = tf.math.log(prob + self.tiny)                              # (batchSz,actionDim)
        return prob, logProb

    def build_critic(self, trainable=True):
        observ = Input(shape=(self.observDim,), dtype=self.tfDtype, name="inputs")
        h = observ
        for ix, units in enumerate(self.critic_hiddenUnits):
            h = self.dense_or_batchNorm(units, "relu", trainable=trainable, name=f"hidden_{ix}")(h)
        action = self.dense_or_batchNorm(self.actionDim, "linear", use_bias=False, trainable=trainable, name="Qs")(h)

        net = Model(inputs=observ, outputs=action, name="critic")
        return net

    @tf.function
    def update_actor(self, observ):
        """ Args: observ: shape=(batchSz,observDim) """
        with tf.GradientTape() as tape:
            actionProb, logProb = self.get_actionProb_logActionProb(observ)     # each (batchSz,actionDim)
            Q1 = self.critic1(observ)                                           # (batchSz,actionDim)
            Q2 = self.critic2(observ)
            Q_min = tf.minimum(Q1, Q2)
            Q_soft = Q_min - self.alpha * logProb                           # (batchSz,actionDim)
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
            target_Q2 = self.target_critic2(next_observ)
            target_Q_min = tf.minimum(target_Q1, target_Q2)
            target_Q_soft = target_Q_min - self.alpha * next_logProb                # (batchSz,actionDim)
            target_V_soft = tf.reduce_sum(target_Q_soft * next_actionProb, axis=1, keepdims=True) # (batchSz,1) 
            y = reward + (1.0 - done) * self.gamma * target_V_soft                  # (batchSz,1)

            Q1 = self.critic1(observ)                                                   # (batchSz,actionDim)
            Q1_selectedByAction = tf.reduce_sum(Q1 * action, axis=1, keepdims=True)     # action as mask; (batchSz,1)
            td_error1 = tf.square(y - Q1_selectedByAction)                              # (batchSz,1)
            td_error1 = importance_weights * td_error1 if self.isPER else td_error1
            critic1_loss = tf.reduce_mean(td_error1)                                    # ()

            Q2 = self.critic2(observ)
            Q2_selectedByAction = tf.reduce_sum(Q2 * action, axis=1, keepdims=True) # action as mask; (batchSz,1)
            td_error2 = tf.square(y - Q2_selectedByAction)
            td_error2 = importance_weights * td_error2 if self.isPER else td_error2
            critic2_loss = tf.reduce_mean(td_error2)

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
            #   td_error = tf.minimum(td_error1, td_error2)             # (batchSz,1)
        return critic1_loss, critic2_loss, td_error1

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: shape=(observDim)
        return:
            action: shape=(observDim)
        """
        if self.explorer.isReadyToExplore():
            action = actionCoder.random_encoded()
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)                     # (1,observDim) to input to net
            actionProb, _ = self.get_actionProb_logActionProb(observ)   # (batchSz,actionDim)
            actionProb = actionProb[0]                                  # (actionDim)
            prob = actionProb.numpy()
            prob[0] = abs(prob[0] + 1 - np.sum(prob))  # for case when tiny residue appears; to make sum(prob)=1; abs for -0
            idx = np.random.choice(self.actionDim, p=prob)              # index of actionToEnv; ()
            action = np.array([1 if i == idx else 0 for i in range(self.actionDim)])  # one-hot; (actionDim)
        return action

