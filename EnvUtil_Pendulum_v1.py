
import numpy as np
import random
from enum import IntEnum
from abc import ABC, abstractmethod
import gymnasium as gym
from NodeCode import NodeCode
from EnvUtil import EnvUtil

class ObservCoder_Pendulum_v1(NodeCode):
    idx = IntEnum('Idx', [
            ('xCoordinate', 0),
            ('yCoordinate', 1),
            ('angularVelocity', 2) 
        ]
    )

    def __init__(self):
        super().__init__()
        self.nNodes = [1, 1, 1]   # 1: continuous; '> 1': discretes which are represented as one-hot vectors
        self.nParameters = len(self.nNodes)

        self.low = np.zeros(self.nParameters, dtype=np.float32)
        self.high = np.zeros(self.nParameters, dtype=np.float32)
        (self.low[self.idx.xCoordinate], self.high[self.idx.xCoordinate]) = (-1, 1)
        (self.low[self.idx.yCoordinate], self.high[self.idx.yCoordinate]) = (-1, 1)
        (self.low[self.idx.angularVelocity], self.high[self.idx.angularVelocity]) = (-8, 8)
        self.spaceType = gym.spaces.Box


class ActionCoder_Pendulum_v1(NodeCode):
    idx = IntEnum('Idx', [
            ('torque', 0)
        ]
    )

    def __init__(self):
        super().__init__()
        self.nNodes = [1]   # 1: continuous; '> 1': discretes which are represented as one-hot vectors
        self.nParameters = len(self.nNodes)

        self.low = np.zeros(self.nParameters, dtype=np.float32)
        self.high = np.zeros(self.nParameters, dtype=np.float32)
        (self.low[self.idx.torque], self.high[self.idx.torque]) = (-2, 2)
        self.spaceType = gym.spaces.Box


class EnvUtil_Pendulum_v1(EnvUtil):
    observCoder = ObservCoder_Pendulum_v1()
    actionCoder = ActionCoder_Pendulum_v1()

