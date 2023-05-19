
import numpy as np
from NodeCode import NodeCode

class EnvUtil:
    observCoder : NodeCode
    actionCoder : NodeCode

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

