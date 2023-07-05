import math

import numpy as np

from .base_type import *


@dataclass
class Action(BaseTypeClass):
    AC_1F: float = 0.
    AC_2F: float = 0.
    AC_3F: float = 0.
    AC_4F: float = 0.
    AC_5F: float = 0.


@dataclass
class Control(BaseTypeClass):
    AC_1F: int = 0
    AC_2F: int = 0
    AC_3F: int = 0
    AC_4F: int = 0
    AC_5F: int = 0

    def to_numpy(self):
        return np.array(list(self.get_values()), dtype=int)


class ActionObj:
    def __init__(self, config: dict):
        self.config = config

        action = Action()
        self.n_elements = len(action)  # num of elements

        n_AC_list = self.config['ACTION_SPACE']
        self.low = np.zeros(self.n_elements)
        self.high = np.array(n_AC_list)

    def get_instance(self):
        return Action()

    def get_action_size(self) -> int:
        return self.n_elements

    def get_action_space(self, bound='low') -> np.ndarray:
        if bound == 'low':
            return self.low
        elif bound == 'high':
            return self.high
        else:
            raise ValueError
        
    def get_random_action(self) -> np.array:
        action = [(np.random.uniform(self.low[i], self.high[i])) for i in range(self.n_elements)]
        return np.array(action)

    def scale_action(self, action) -> np.array:
        """
        action = [-1, -1, -1, -1, -1]
        dimension 별로 action space에 맞게 unnormalize (decoding)
        
        - 1F: +1 * 2.5 -> [0.5]
        - 2F: +1 * 3.5 -> [0, 7]
        - 3F: +1 * 2.5 -> [0, 5]
        - 4F: +1 * 2.5 -> [0, 5]
        - 5F: +1 * 2.5 -> [0, 5]
        """
        return (action + 1) * self.high / 2


class ControlObj:
    def __init__(self, config: dict):
        self.config = config
        
        control = Control()
        self.n_elements = len(control)  # num of elements
    
    def get_instance(self):
        return Control()

    def custom_round(self, value):
        if value - math.floor(value) > self.config['ROUND_THRESHOLD']:
            return math.ceil(value)
        else:
            return math.floor(value)

    def control_fromAction(self, action: np.array) -> Control:
        """
        Args:
            action (np.array)

        Retruns:
            control (Control)
        """
        control = Control(
            AC_1F = self.custom_round(action[0]),
            AC_2F = self.custom_round(action[1]),
            AC_3F = self.custom_round(action[2]),
            AC_4F = self.custom_round(action[3]),
            AC_5F = self.custom_round(action[4])
        )
        return control

    def control_fromData(self, data: dict) -> Control:
        """
        Args:
            data (dict): a dictionary made from a row of data dataframe

        Retruns:
            control (Control)
        """
        control = Control(
            AC_1F = data['AC_1F'],
            AC_2F = data['AC_2F'],
            AC_3F = data['AC_3F'],
            AC_4F = data['AC_4F'],
            AC_5F = data['AC_5F']
        )
        return control
