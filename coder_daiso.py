
import numpy as np
import random
from coder import NodeCoder
import os
import json

class NodeCoder_daiso:
    def __init__(self, nNodes=None, low=None, high=None, scaleshift=None, possibles=None, isDecodedScalar=False):
        """
        Args:
            nNodes: a vector that contains the number of nodes needed for each parameter in the given vector; 1D ndarray.
                    Like [1, 3] where 1 is for a continuous and 3 for a discrete parameter.
            low, high: lower and upper bound of countinuous parameters. zeros for discrete parameters; each 1D ndarray.
                    Like [-10, 0] and [10, 0].
            scaleshift: scale and shift mode for continuous parameters. Among {None,'unit','sym_unit'} 
                    where None is for no change, 'unit' is into [0,1], 'sym_unit' is into [-1,1]; str 
            possibles: list that contains lists of possible values for discrete parameters. [] for continuous parameters.
                    Like [[], [5, 8, 3]].
            isDecodedScalar: if True, the decoded one is a scalar; special cases like CartPole-v1
        """
        self.nNodes = [] if nNodes is None else nNodes                                # 1: continuous; '> 1': discretes 
        assert all(n >= 1 and n % 1 == 0 for n in self.nNodes), \
                f"nNodes={nNodes} are expected to contain integers larger than 0"
        self.nParameters = len(self.nNodes)
        self.encodedDim = sum(self.nNodes)  # dimension of encoded == num of nodes for all parameters
        self.isDecodedScalar = isDecodedScalar
   
        # continuous 
        self.low = np.zeros(self.nParameters, dtype=np.float32) if low is None else np.array(low)    # 'is' instead of '=='
        self.high = np.zeros(self.nParameters, dtype=np.float32) if high is None else np.array(high) # 'is' instead of '=='
        self.scaleshift = scaleshift

        # dicrete
        self.possibles = [[] for i in range(self.nParameters)] if possibles is None else possibles   # 'is' instead of '=='

        with open(os.getcwd()+'/env_daiso/config.json') as f:
            env_config = json.load(f)

        self.start_time = env_config['RL_START_TIME']
        self.unit_timestep = env_config['unit_timestep']

        agentMode = env_config["AgentMode"]
        self.comfortLow, self.comfortHigh = {}, {}
        for floor, value in env_config["COMFORT_ZONE"][agentMode].items():
            low, high = value
            self.comfortLow[floor] = low
            self.comfortHigh[floor] = high

    def encode(self, vec):
        """ Get nodeCode from a vector vec. 
        Args:
            vec: an environment_vector as an array of parameters; 1D ndarray 
        Returns:
            nodeVec: encoded neuralnet_vector from vec; 1D ndarray
        """
        nodeVec = [ 
            vec[0],
            (vec[1] - (60 * self.start_time)) // self.unit_timestep,
            vec[2] - self.comfortLow['1F'],
            vec[3] - self.comfortLow['2F'],
            vec[4] - self.comfortLow['3F'],
            vec[5] - self.comfortLow['4F'],
            vec[6] - self.comfortLow['5F'],
            vec[2] - self.comfortHigh['1F'],
            vec[3] - self.comfortHigh['2F'],
            vec[4] - self.comfortHigh['3F'],
            vec[5] - self.comfortHigh['4F'],
            vec[6] - self.comfortHigh['5F'],
            vec[2] - vec[7],
            vec[3] - vec[7],
            vec[4] - vec[7],
            vec[5] - vec[7],
            vec[6] - vec[7],
            vec[8],
            vec[9],
            vec[10],
            vec[11],
            vec[12]
        ]

        nodeVec = np.array(nodeVec)
        return nodeVec

    def decode(self, nodeVec):
        """ Get vec from nodeVec. 
        Args:
            nodeVec: a neuralnet_vector; 1D ndarray
        Returns:
            vec: an environment_vector as an array of parameters; 1D ndarray 
        """
            #   print(f"nodeVec={nodeVec}")
        vec = []
        nodeIdx = 0
        for ix in range(len(self.nNodes)):
            if self.nNodes[ix] == 1:  # continuous
                if self.scaleshift == 'sym_unit':
                    vec.append(((nodeVec[ix] + 1) / 2) * (self.high[ix] - self.low[ix]) + self.low[ix])  # from range (-1,1) to (low,high)
                elif self.scaleshift == 'unit':
                    vec.append(nodeVec[ix] * (self.high[ix] - self.low[ix]) + self.low[ix])              # from range (0,1) to (low,high)
                else:
                    vec.append(nodeVec[ix])
                nodeIdx += 1
            else:  # self.nNodes[ix] > 1; discrete
                oneHot = nodeVec[nodeIdx : nodeIdx + self.nNodes[ix]]
                    #   print(f"in decode:\noneHot={oneHot}")
                    #   print(f"type(oneHot)={type(oneHot)}")
                    #   print(f"type(oneHot[0])={type(oneHot[0])}")
                idx = np.where(oneHot == 1)[0][0]
                vec.append(self.possibles[ix][idx]) 
                nodeIdx += self.nNodes[ix]

        vec = np.array(vec)
        if self.isDecodedScalar:  
            vec = vec[0]    # vec is a scalar
        return vec 

    def random_decoded(self):
        """ returns a random environment_vector """
        vec = []
            #   print(f"self.nNodes={self.nNodes}") 
        for ix in range(len(self.nNodes)):
            if self.nNodes[ix] == 1:  # continuous
                vec.append(random.uniform(self.low[ix], self.high[ix]))
                    #   print(f"vec={vec}")
            else:  # self.nNodes[ix] > 1; discrete
                vec.append(random.choice(self.possibles[ix])) 
                    #   print(f"vec={vec}")
        vec = np.array(vec)
            #   print(f"vec={vec}")

        return vec


class Coder_daiso:
    """ 
    encode: neuralnet side => environment side 
    decode: environment side => neuralnet side 
    """
    def __init__(self, envName, config, logger):
        self.logger = logger
        observ_config = config[envName]["observ"]
        action_config = config[envName]["action"]
        self.observCoder = NodeCoder_daiso(nNodes = observ_config["nNodes"], 
                                     low = observ_config["low"], 
                                     high = observ_config["high"], 
                                     possibles = observ_config["possibles"], 
                                     scaleshift = observ_config["scaleshift"], 
                                     isDecodedScalar = observ_config["isDecodedScalar"])
        self.actionCoder = NodeCoder(nNodes = action_config["nNodes"], 
                                     low = action_config["low"], 
                                     high = action_config["high"], 
                                     possibles = action_config["possibles"], 
                                     scaleshift = action_config["scaleshift"], 
                                     isDecodedScalar = action_config["isDecodedScalar"])

    def experienceFrom(self, observFrEnv, actionToEnv, reward, next_observFrEnv, done, npDtype):
        observ = self.observCoder.encode(observFrEnv)
        next_observ = self.observCoder.encode(next_observFrEnv)
        action = self.actionCoder.encode(actionToEnv)    # actionToEnv: scalar, action: node_vector
        ex = (
                np.array(observ, dtype=npDtype),        # (observDim)
                np.array(action, dtype=npDtype),        # (actionDim)
                np.array([reward], dtype=npDtype),      # scalar to ndarray of dtype
                np.array(next_observ, dtype=npDtype),   # (observDim)
                np.array([done], dtype=npDtype)         # bool to ndarray of dtype
        )
        return ex

