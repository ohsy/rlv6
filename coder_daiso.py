
import numpy as np
import random
from coder import NodeCoder
import os
import json
from coder import NodeCoder, Coder
            

class NodeCoder_daiso_observ(NodeCoder):
    def __init__(self, nNodes=None, low=None, high=None, scaleshift=None, possibles=None, isDecodedScalar=False, isStochastic=False):
        super().__init__(nNodes, low, high, scaleshift, possibles, isDecodedScalar, isStochastic)
        
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
            vec[0],  # days
            (vec[1] - (60 * self.start_time)) // self.unit_timestep,  # timestep from minutes
            vec[2] - self.comfortLow['1F'],  # from T_ra_1F
            vec[3] - self.comfortLow['2F'],  # from T_ra_2F
            vec[4] - self.comfortLow['3F'],  # from T_ra_3F
            vec[5] - self.comfortLow['4F'],  # from T_ra_4F
            vec[6] - self.comfortLow['5F'],  # from T_ra_5F
            vec[2] - self.comfortHigh['1F'],
            vec[3] - self.comfortHigh['2F'],
            vec[4] - self.comfortHigh['3F'],
            vec[5] - self.comfortHigh['4F'],
            vec[6] - self.comfortHigh['5F'],
            vec[2] - vec[7],  # from T_ra_1F and T_oa
            vec[3] - vec[7],  # from T_ra_2F and T_oa
            vec[4] - vec[7],  # from T_ra_3F and T_oa
            vec[5] - vec[7],  # from T_ra_4F and T_oa
            vec[6] - vec[7],  # from T_ra_5F and T_oa
            vec[8],  # T_oa_min
            vec[9],  # T_oa_max
            vec[10],  # CA
            vec[11],  # n_HC_instant
            vec[12]  # n_HC
        ]

        nodeVec = np.array(nodeVec)
        return nodeVec

class NodeCoder_daiso_action(NodeCoder):
    def decode(self, nodeVec):
        """ Get vec from nodeVec. 
        Args:
            nodeVec: a neuralnet_vector; 1D ndarray
        Returns:
            vec: an environment_vector as an array of parameters; 1D ndarray 
        """
        vec = []
        nodeIdx = 0
            #   print(f"nodeVec={nodeVec}")
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
                    #   oneHot = nodeVec[nodeIdx : nodeIdx + self.nNodes[ix]]
                probs = nodeVec[nodeIdx : nodeIdx + self.nNodes[ix]]
                    #   print(f"in decode:\probs={probs}")
                    #   print(f"type(probs)={type(probs)}")
                    #   print(f"type(probs[0])={type(probs[0])}")
                if self.isStochastic:
                    probs = np.float64(probs)  # since multinomial casts by float64 before checking if sum of probs is 1.0
                    residue = 1.0 - np.sum(probs)
                    if residue != 0.0:
                        maxIdx = np.argmax(probs)
                        probs[maxIdx] += residue
                    nTrials = 1  # for self.nNodes[ix]; 1 is very stochastic; as it grows bigger, more deterministic
                    dist = np.random.multinomial(nTrials, probs)  # one-hot vector like [0, 1, 0, 0, 0] for nTrials=1
                    idx = np.argmax(dist)
                else:
                    idx = np.argmax(probs)

                    #   print(f"idx={idx}")
                vec.append(self.possibles[ix][idx])
                nodeIdx += self.nNodes[ix]

        vec = np.array(vec)
        if self.isDecodedScalar:
            vec = vec[0]    # vec is a scalar
        return vec


class Coder_daiso(Coder):
    """ 
    encode: neuralnet side => environment side 
    decode: environment side => neuralnet side 
    """
    def __init__(self, config, envName, agentName, logger):
        self.logger = logger
        observ_config = config[envName]["observ"]
        action_config = config[envName]["action"]
        agent_config = config[agentName]
        self.observCoder = NodeCoder_daiso_observ(
                nNodes = observ_config["nNodes"], 
                low = observ_config["low"], 
                high = observ_config["high"], 
                possibles = observ_config["possibles"], 
                scaleshift = observ_config["scaleshift"], 
                isDecodedScalar = observ_config["isDecodedScalar"])
        self.actionCoder = NodeCoder_daiso_action(
                nNodes = action_config["nNodes"], 
                low = action_config["low"],
                high = action_config["high"], 
                possibles = action_config["possibles"], 
                scaleshift = action_config["scaleshift"], 
                isDecodedScalar = action_config["isDecodedScalar"],
                isStochastic = agent_config["isActionStochastic"] if "isActionStochastic" in agent_config else config["isActionStochastic"]) 

