"""
2023.04.25 made by soh

- Made from ddpgInGym.py using TF2 and keras.
- Losses diverge.
- At first, comment out all of '@tf.function's
- Problem: keep diverging
=> Fixed: in soft_update(): actor.variables and critic.variables are zipped together in zip() 
    though their lengths are different 
    => separated. 
- It seems that use_bias=False in output layer in actor makes training faster.

- 1 actor, 1 critic, 1 target actor, 1 target critic: usually, average reward in episode can be larger than -0.005 
    within about 100 episodes.
- 1 actor, 2 critic, 1 target actor, 2 target critic: usually, average reward in episode can be larger than -0.005 
    within about 100 episodes. 
- 1 actor, 2 critic, 2 target critic: usually, average reward in episode can be larger than -0.005 
    in about 200 episodes or never. 
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

2023.07.26

- target actor is removed. critic2 is always used.
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
from Agent import Agent
from replaymemory import ReplayMemory, PERMemory
from importlib import import_module


class ActorCritic(Agent):
    def __init__(self, envName, mode, config, logger, observDim, actionDim):
        super().__init__(envName, mode, config, logger, observDim, actionDim)

        if mode == "train":
            self.actor = self.build_actor()
            self.critic1 = self.build_critic() 
            self.critic2 = self.build_critic()
            self.target_critic1 = self.build_critic(trainable=False)
            self.target_critic2 = self.build_critic(trainable=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            self.critic2_optimizer = Adam(self.critic_lr)
        elif mode == "test": 
            self.actor = load_model(self.savePath_actor, compile=False)
            self.logger.info(f"actor is loaded from {self.savePath_actor}")
            self.actor.summary(print_fn=self.logger.info)
        elif mode == "continued_train":
            self.actor = load_model(self.savePath_actor, compile=False)
            self.critic1 = load_model(self.savePath_critic1, compile=False)
            self.critic2 = load_model(self.savePath_critic2, compile=False)
            self.target_critic1 = load_model(self.savePath_target_critic1, compile=False)
            self.target_critic2 = load_model(self.savePath_target_critic2, compile=False)
            self.actor_optimizer = Adam(self.actor_lr)
            self.critic1_optimizer = Adam(self.critic_lr)
            self.critic2_optimizer = Adam(self.critic_lr)
            self.explorer.load()
            self.actor.summary(print_fn=self.logger.info)
            self.critic1.summary(print_fn=self.logger.info)

    def build_actor(self):
        pass

    def build_critic(self, trainable=True):
        pass

    @tf.function
    def update_actor(self, observ):
        pass

    @tf.function
    def update_critic(self, observ, action, reward, next_observ, done, importance_weights):
        pass


    #   @tf.function  # NOTE: recommended not to use tf.function. zip problem?? And not much enhancement. 
    def soft_update(self):
        for target_param1, param1 in zip(self.target_critic1.variables, self.critic1.variables):
            target_param1.assign(target_param1 * (1.0 - self.tau) + param1 * self.tau)
        for target_param2, param2 in zip(self.target_critic2.variables, self.critic2.variables):
            target_param2.assign(target_param2 * (1.0 - self.tau) + param2 * self.tau)

    #   @tf.function  # NOTE recommended not to use tf.function. cross function problem?? And not much enhancement.
    def train(self, batch, importance_weights):
        critic1_loss, critic2_loss, td_error1 = self.update_critic(
                batch.observ,
                batch.action,
                batch.reward,
                batch.next_observ,
                batch.done,
                importance_weights
        )
        actor_loss = self.update_actor(batch.observ)
        self.soft_update()
        return (critic1_loss, actor_loss), td_error1  # (,) to be consistent with DQN return

    #   @tf.function  # NOTE recommended not to use tf.function. And not much enhancement.
    def act(self, observ, actionCoder):
        """
        Args:
            observ: 1d ndarray of shape=(observDim)
        return:
            action: 1d ndarray of shape=(actionDim)
        """
        if self.explorer.isReadyToExplore():
            action = actionCoder.random_encoded()
        else:
            observ = tf.convert_to_tensor(observ)
            observ = tf.expand_dims(observ, axis=0)     # (1,observDim) to input to net
            action = self.actor(observ)                 # (1,actionDim)
            action = action[0]                          # (actionDim)
            action = action.numpy()                     # ndarray
        return action                                   

    def save(self):
        self.actor.save(self.savePath_actor)
        self.critic1.save(self.savePath_critic1)
        self.critic2.save(self.savePath_critic2)
        self.target_critic1.save(self.savePath_target_critic1)
        self.target_critic2.save(self.savePath_target_critic2)
        self.replayMemory.save(self.savePath_replayMemory)
        self.explorer.save()

    def summary(self):
        self.actor.summary(print_fn=self.logger.info)   # to print in logger file
        self.critic1.summary(print_fn=self.logger.info) # to print in logger file
        
