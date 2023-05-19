
"""
2023.05.11
"""
import numpy as np
#   import random
from enum import IntEnum
#   from abc import ABC, abstractmethod
import gymnasium as gym
from NodeCode import NodeCode
from EnvUtil import EnvUtil

class ObservCoder_CartPole_v1(NodeCode):
    idx = IntEnum('Idx', [          # NOTE: continuous parameters before discrete
            ('cartPosition', 0),
            ('cartVelocity', 1),
            ('poleAngle', 2),
            ('poleAngularVelocity', 3) 
        ]
    )

    def __init__(self):
        super().__init__()
        self.nNodes = [1, 1, 1, 1]   # 1: continuous; '> 1': discretes which are represented as one-hot vectors
        self.nParameters = len(self.nNodes)
   
        # continuous 
        halfMin = np.finfo(np.float32).min / 2  # '/2' to prevent overflow
        halfMax = np.finfo(np.float32).max / 2  # '/2' to prevent overflow
        self.low = np.zeros(self.nParameters, dtype=np.float32)
        self.high = np.zeros(self.nParameters, dtype=np.float32)
        (self.low[self.idx.cartPosition], self.high[self.idx.cartPosition]) = (-4.8, 4.8)
        (self.low[self.idx.cartVelocity], self.high[self.idx.cartVelocity]) = (halfMin, halfMax)
        (self.low[self.idx.poleAngle], self.high[self.idx.poleAngle]) = (-0.418, 0.418)
        (self.low[self.idx.poleAngularVelocity], self.high[self.idx.poleAngularVelocity]) = (halfMin, halfMax)
        self.spaceType = gym.spaces.Box


class ActionCoder_CartPole_v1(NodeCode):
    idx = IntEnum('Idx', [
            ('pushTo', 0)
        ]
    )

    def __init__(self):
        super().__init__()
        self.nNodes = [2]   # 1: continuous; '> 1': discretes which are represented as one-hot vectors
        self.nParameters = len(self.nNodes)

        # discrete
        self.possibles = [[] for i in range(self.nParameters)]
        self.possibles[self.idx.pushTo] = (0, 1)
        self.n = len(self.possibles[self.idx.pushTo])
        self.spaceType = gym.spaces.Discrete
        #   self.nodeCode = NodeCode(nNodes, possibles=possibles)

    def decode(cls, action):
            #   print(f"in decode:\naction={action}")
        actionToEnv = super().decode(action)  
            #   print(f"actionToEnv={actionToEnv}")
        actionToEnv = actionToEnv[0]    # CartPole-v1 actionToEnv shape=()
        return actionToEnv

    def encode(cls, actionToEnv):
        actionToEnv = [actionToEnv]     # CartPole-v1 actionToEnv shape=()
        action = super().encode(actionToEnv)
        return action

    """
    @classmethod
    def random_vec(cls):
        return cls.nodeCode.random_vec()
    """

class EnvUtil_CartPole_v1(EnvUtil):
    observCoder = ObservCoder_CartPole_v1()
    actionCoder = ActionCoder_CartPole_v1()

