
"""
2023.05.12
- A new format NodeCode aka node code.

    NodeCode encodes ordinary_vectors into node_vectors and decodes node_vector backs into ordinary_vectors.
    An ordinary_vector is a list or array composed of continuous or discrete parameters.
    A node_vector is a list or array where a discrete parameter is represented as a local one-hot vector.
    If there are multiple discrete parameters, their one-hot vectors are concatenated to form a node_vector.

    It can be used for neuralnets as observ or action.
    A continuous parameter needs one node in observation layer of neuralnets. 
    (Actually two nodes of mean and stddev are needed for action parameters, but let's call it one node
    to make 1 in the list represent a continuous parameter.)
    A discrete parameter is represented as a one-hot vector and needs nodes as many as 'possibles'. 
    nNodes is a list of which an element is the number of nodes needed in neuralnet for a parameter.
    AS a discrete parameter needs more than one node, 1 in the list indicates a continuous parameter 
    while an integer larger than 1 indicates a discrete parameter.

    The corresponding vector is called 'vec' and contains continuous or discrete parameters.
    A continuous parameter has a number value in the range ('low', 'high') like Box of Gym Environment.
    A discrete parameter has an number value among a list of 'possibles' like Discrete of Gym Environment.
"""
import numpy as np
import random
#   from enum import IntEnum
#   from abc import ABC, abstractmethod

class NodeCode:
    def __init__(self, nNodes=None, low=None, high=None, scaleshift=None, possibles=None):
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
        """
        self.nNodes = [] if nNodes is None else nNodes                                # 1: continuous; '> 1': discretes 
        assert all(n >= 1 and n % 1 == 0 for n in self.nNodes), f"nNodes={nNodes} are expected to contain integers larger than 0"
        self.nParameters = len(self.nNodes)
   
        # continuous 
        self.low = np.zeros(self.nParameters, dtype=np.float32) if low is None else low              # 'is' instead of '=='
        self.high = np.zeros(self.nParameters, dtype=np.float32) if high is None else high           # 'is' instead of '=='
        self.scaleshift = scaleshift

        # dicrete
        self.possibles = [[] for i in range(self.nParameters)] if possibles is None else possibles   # 'is' instead of '=='

    def encode(self, vec):
        """ Get nodeCode from a vector vec. 
        Args:
            vec: an ordinary_vector as an array of parameters; 1D ndarray 
        Returns:
            nodeCode: encoded node_vector from vec; 1D ndarray
        """
        nodeVec = []
        for ix in range(0, len(self.nNodes)):
            if self.nNodes[ix] == 1:  # continuous
                if self.scaleshift == 'sym_unit':
                    nodeVec.append((vec - self.low) / (self.high - self.low) * 2 - 1)    # from range (low,high) to (-1,1)
                elif self.scaleshift == 'unit':
                    nodeVec.append((vec - self.low) / (self.high - self.low) - 0.5)      # from range (low,high) to (0,1)
                else:
                    nodeVec.append(vec[ix]) 
            else:  # self.nNodes[ix] > 1; discrete
                idx = self.possibles[ix].index(vec[ix])
                nodeVec += [1 if i == idx else 0 for i in range(self.nNodes[ix])]  # one-hot vector

        nodeVec = np.array(nodeVec)
        return nodeVec

    def decode(self, nodeVec):
        """ Get vec from nodeVec. 
        Args:
            nodeVec: a node_vector; 1D ndarray
        Returns:
            vec: an ordinary_vector as an array of parameters; 1D ndarray 
        """
            #   print(f"nodeVec={nodeVec}")
        vec = []
        nodeIdx = 0
        for ix in range(0, len(self.nNodes)):
            if self.nNodes[ix] == 1:  # continuous
                if self.scaleshift == 'sym_unit':
                    vec.append(((action + 1) / 2) * (self.high - self.low) + self.low)  # from range (-1,1) to (low,high)
                elif self.scaleshift == 'unit':
                    vec.append(action * (self.high - self.low) + self.low)              # from range (0,1) to (low,high)
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
        return vec 

    def random_vec(self):
        """ returns a random ordinary_vector """
        vec = []
            #   print(f"self.nNodes={self.nNodes}") 
        for ix in range(0, len(self.nNodes)):
            if self.nNodes[ix] == 1:  # continuous
                vec.append(random.uniform(self.low[ix], self.high[ix]))
                    #   print(f"vec={vec}")
            else:  # self.nNodes[ix] > 1; discrete
                vec.append(random.choice(self.possibles[ix])) 
                    #   print(f"vec={vec}")
        vec = np.array(vec)
            #   print(f"vec={vec}")

        return vec

