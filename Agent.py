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
from replaybuffer import ReplayBuffer, PERBuffer
from importlib import import_module


class Agent:
    def __init__(self, envName, mode, config, logger):
        self.mode = mode  # config["Mode"]
        self.config = config
        self.logger = logger
        self.npDtype = np.dtype(config["dtype"])
        self.tfDtype = tf.convert_to_tensor(np.zeros((1,), dtype=self.npDtype)).dtype  # tf doesn't have dtype()

        explorer = config["Explorer"]
        self.alpha = config["TemperatureParameter_alpha"]
        self.isActionStochastic = config["isActionStochastic"]  # vs. deterministic with max prob.
        if self.__class__.__name__ in config:
            agent_config = config[self.__class__.__name__] 
            explorer = agent_config["Explorer"] 
            self.alpha = agent_config["TemperatureParameter_alpha"] 
            self.isActionStochastic = config["isActionStochastic"]  
        self.alpha = tf.Variable(self.alpha, dtype=self.tfDtype)  

        self.isTargetActor = config["TargetActor"]
        self.isCritic2 = config["Critic2"]
        self.savePath = f"{config['SavePath']}/{envName}/{self.__class__.__name__}"
        self.writer = tf.summary.create_file_writer(config["SummaryWriterPath"])
        self.isRewardNorm = config["RewardNormalization"]
        self.isPER = config["PER"]

        self.memoryCapacity = config[envName]["MemoryCapacity"]
        self.replayBuffer = PERBuffer(mode, self.memoryCapacity, self.isRewardNorm, self.npDtype) if self.isPER \
                       else ReplayBuffer(mode, self.memoryCapacity, self.isRewardNorm, self.npDtype)

        explorerModule = import_module(f"explorer")
        Explorer = getattr(explorerModule, f"Explorer_{explorer}")
        self.explorer = Explorer(mode, config, self.savePath, self.replayBuffer)
        self.memoryCnt_toStartTrain = self.explorer.get_memoryCnt_toStartTrain()

        if self.isRewardNorm:
            # self.recentMemoryCapacity = self.memoryCapacity // 4
            # if "train" in self.mode:
            #     self.reward_memory = deque(maxlen=self.memoryCapacity // 2)
            #     self.recent_reward = deque(maxlen=self.recentMemoryCapacity)

            self.reward_norm_steps = 200
            self.reward_mean = 1
            # self.rewardNormalizationThreshold = 0.1  # begin reward norm after buffer is filled over threshold
            self.rewardNormalizationThreshold = 0.7

        self.batchSz = config["BatchSize"]
        self.tau = config["SoftUpdateRate_tau"] 
        self.gamma = tf.Variable(config["RewardDiscountRate_gamma"], dtype=self.tfDtype)
        self.tiny = 1e-6  # added to denominator to prevent inf; NOTE: value < 1e-6 (like 1e-7) is considered as 0

        self.batchNormInUnitsList = config["BatchNorm_inUnitsList"]  # to represent batchNorm in XXX_units list 'bn'

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

    def isReadyToTrain(self):
        b1 = self.mode in ["train", "continued_train"]
        b2 = self.replayBuffer.memoryCnt > self.memoryCnt_toStartTrain
        b3 = self.replayBuffer.memoryCnt > self.batchSz
        #   b4 = self.actCnt % 1 == 0
        return b1 and b2 and b3  #   and b4

    def summaryWrite(self, key, value, step):
        with self.writer.as_default():
            tf.summary.scalar(key, value, step=step)

