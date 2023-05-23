
import numpy as np
from NodeCoder import NodeCoder

class Coder:
    observCoder : NodeCoder
    actionCoder : NodeCoder

    @classmethod
    def experienceFrom(cls, observFrEnv, actionToEnv, reward, next_observFrEnv, done, npDtype):
        observ = cls.observCoder.encode(observFrEnv)
        next_observ = cls.observCoder.encode(next_observFrEnv)
        action = cls.actionCoder.encode(actionToEnv)    # actionToEnv: scalar, action: node_vector
        ex = (
                np.array(observ, dtype=npDtype),        # already ndarray but change dtype
                np.array(action, dtype=npDtype),        # already ndarray but change dtype 
                np.array([reward], dtype=npDtype),      # scalar to ndarray of dtype
                np.array(next_observ, dtype=npDtype),   # already ndarray but change dtype  
                np.array([done], dtype=npDtype)         # bool to ndarray of dtype
        )
        return ex


class Coder_CartPole_v1(Coder):
    observCoder = NodeCoder(nNodes=[1,1,1,1],
                           low=[-4.8, np.finfo(np.float32).min / 2, -0.418, np.finfo(np.float32).min / 2],
                           high=[4.8, np.finfo(np.float32).max / 2, 0.418, np.finfo(np.float32).max / 2], scaleshift=None)
    actionCoder = NodeCoder(nNodes=[2], possibles=[[0,1]], scaleshift=None, isDecodedScalar=True)


class Coder_Pendulum_v1(Coder):
    observCoder = NodeCoder(nNodes=[1,1,1], low=[-1, -1, -8], high=[1,1,8], scaleshift=None)
    actionCoder = NodeCoder(nNodes=[1], low=[-2], high=[2], scaleshift=None)

