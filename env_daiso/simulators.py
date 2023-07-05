import joblib
from abc import *

import numpy as np
import pandas as pd
import tensorflow as tf

from env_daiso.stateobservation_DIS import State
from env_daiso.dynamic_modules.makeDF import calDfs, eachFloorE, returnDeltaT


class SimulatorBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _preprocessing(self):
        # data preprocessing method
        pass

    @abstractmethod
    def simulate(self):
        # simulate with input data called from outside
        pass


class HCSimulator(SimulatorBase):
    def __init__(self, model_path: str) -> None:
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(f'{model_path}_scaler.pkl')

    def _preprocessing(self, data, past_n_HC):
        inputs = np.array([
            data.T_oa,
            data.CA,
            data.days,
            past_n_HC[0],
            past_n_HC[1],
            past_n_HC[2],
            data.minutes,
        ]).reshape((1, -1))
        return self.scaler.transform(inputs)

    def simulate(self, data, past_n_HC):
        """
        Args:
            data (dict): data should contain
            -> TA, CA, days, instant_head[t-1], instant_head[t-2], instant_head[t-3], hour, min
        """
        inputs = self._preprocessing(data, past_n_HC)
        return float(self.model(inputs, training=False).numpy().reshape(1)[0])


class MlpRASimulator(SimulatorBase):
    def __init__(self, model_path: str) -> None:
        self.models, self.scalers = [], []
        for i in range(len(model_path)):
            model_p = model_path[i]
            scaler_p = f'{model_p}_scaler.pkl'

            self.models.append(tf.keras.models.load_model(model_p))
            self.scalers.append(joblib.load(scaler_p))

    def _preprocessing(self, data, E_ehp, i):
        inputs = np.array([
            data.minutes,
            data.T_ra_1F,
            data.T_ra_2F,
            data.T_ra_3F,
            data.T_ra_4F,
            data.T_ra_5F,
            E_ehp[0],
            E_ehp[1],
            E_ehp[2],
            E_ehp[3],
            E_ehp[4],
            data.T_oa,
            data.CA,
            data.n_HC_instant,
            data.n_HC,
            data.days
        ]).reshape((1, -1))
        return self.scalers[i].transform(inputs)
    
    def simulate(self, data: State, E_ehp: np.array):
        """
        Args:
            data
            -> minutes, 1F_temp, 2F_temp, 3F_temp, 4F_temp, 5F_temp, demand1, demand2, demand3, demand4, demand5, TA, CA, instant_head, cumul_head, days
        Returns:
            delta_Ts (ndarray): delta_T_ra of each floor
        """
        delta_Ts = []
        for i in range(len(self.models)):
            inputs = self._preprocessing(data, E_ehp, i)
            delta_T = self.models[i](inputs, training=False).numpy().reshape(1)[0]
            delta_Ts.append(delta_T)
        return np.array(delta_Ts)


class LatticeRASimulator(SimulatorBase):
    def __init__(self, model_path: str) -> None:
        self.models, self.scalers = [], []
        for i in range(len(model_path)):
            model_p = model_path[i]
            self.models.append(tf.keras.models.load_model(model_p))

    def _preprocessing(self, data, E_ehp):
        inputs = [
            tf.convert_to_tensor([data.minutes]),
            tf.convert_to_tensor([data.T_ra_1F]),
            tf.convert_to_tensor([data.T_ra_2F]),
            tf.convert_to_tensor([data.T_ra_3F]),
            tf.convert_to_tensor([data.T_ra_4F]),
            tf.convert_to_tensor([data.T_ra_5F]),
            tf.convert_to_tensor([E_ehp[0]]),
            tf.convert_to_tensor([E_ehp[1]]),
            tf.convert_to_tensor([E_ehp[2]]),
            tf.convert_to_tensor([E_ehp[3]]),
            tf.convert_to_tensor([E_ehp[4]]),
            tf.convert_to_tensor([data.T_oa]),
            tf.convert_to_tensor([data.CA]),
            tf.convert_to_tensor([data.n_HC_instant]),
            tf.convert_to_tensor([data.n_HC]),
            tf.convert_to_tensor([data.days])
        ]
        return inputs
    
    def simulate(self, data: State, E_ehp: np.array):
        """
        Args:
            data
            -> minutes, 1F_temp, 2F_temp, 3F_temp, 4F_temp, 5F_temp, demand1, demand2, demand3, demand4, demand5, TA, CA, instant_head, cumul_head, days
        Returns:
            delta_Ts (ndarray): delta_T_ra of each floor
        """
    
        delta_Ts = []
        for i in range(len(self.models)):
            inputs = self._preprocessing(data, E_ehp)
            delta_T = self.models[i](inputs, training=False).numpy().reshape(1)[0]
            delta_Ts.append(delta_T)
        return np.array(delta_Ts)


class DynamicRASimulator(SimulatorBase):
    def __init__(self, model_path) -> None:
        pass

    def predict(self, inputs):
        '''
        df = pd.DataFrame(columns=['hour', 'minute',
                                    '1F_temp','2F_temp','3F_temp','4F_temp','5F_temp',
                                    'demand1','demand2','demand3','demand4','demand5',
                                    'TA','CA','instant_head','cumul_head','days'])
        '''
        E_df = calDfs(inputs)
        floorE_df = eachFloorE(E_df)
        delt_df = returnDeltaT(floorE_df)
        delt_array = delt_df.to_numpy().reshape(-1)
        return delt_array

    def _preprocessing(self, data, E_ehp):
        hour = data.minutes // 60
        minute = data.minutes - (hour * 60)

        inputs = np.array([
            hour,
            minute,
            data.T_ra_1F,
            data.T_ra_2F,
            data.T_ra_3F,
            data.T_ra_4F,
            data.T_ra_5F,
            E_ehp[0],
            E_ehp[1],
            E_ehp[2],
            E_ehp[3],
            E_ehp[4],
            data.T_oa,
            data.CA,
            data.n_HC_instant,
            data.n_HC,
            data.days
        ]).reshape((1, -1))
        return pd.DataFrame(inputs, columns=['hour', 'minute',
                                            '1F_temp','2F_temp','3F_temp','4F_temp','5F_temp',
                                            'demand1','demand2','demand3','demand4','demand5',
                                            'TA','CA','instant_head','cumul_head','days'])
    
    def simulate(self, data: State, E_ehp: np.array):
        inputs = self._preprocessing(data, E_ehp)
        return self.predict(inputs)
