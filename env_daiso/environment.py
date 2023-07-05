"""
Environment for Daiso Project

2023.05 sangjin
"""

import os
import json
import numpy as np

from env_daiso.simulators import HCSimulator
from env_daiso.estimator_DIS import Estimator
from env_daiso.data_manager import DataManager
from env_daiso.actioncontrol_DIS import ControlObj, ActionObj
from env_daiso.stateobservation_DIS import StateObj, ObservationObj, State


class DaisoSokcho:
    def __init__(self, phase="train"):
        self.phase = phase
        self.action_space = None
        self.observation_space = None
       
        with open(os.getcwd() + "/env_daiso/config.json") as f:
            config = json.load(f) 
        self.config = config

        self.data_manager = DataManager(config, phase)

        self.actionObj = ActionObj(config)
        self.ctrlObj = ControlObj(config)
        self.stateObj = StateObj(config)
        self.obsObj = ObservationObj(config)

        self.estimator = Estimator(config["SEASON"], self.stateObj)

        self.unit_timestep = config['unit_timestep']
        self.ehp = config['EHP']

        self.comfortLows, self.comfortHighs = self.obsObj.get_comfortZone()

        """ Load simulator and input data scaler """
        simulator_model = f'simulator_RA_{config["simulator_RA_model"]}'
        simulator_pre_model = f'simulator_pre_RA_{config["simulator_RA_model"]}'

        ROOT_PATH = config['ROOT_PATH']
        if 'mlp' in simulator_model:
            from env_daiso.simulators import MlpRASimulator as RASimulator
            RA_model_path = [os.path.join(ROOT_PATH, config[simulator_model][f'{i+1}F']) for i in range(len(config[simulator_model]))]
            pre_RA_model_path = [os.path.join(ROOT_PATH, config[simulator_pre_model][f'{i+1}F']) for i in range(len(config[simulator_pre_model]))]
        elif 'lattice' in simulator_model:
            from env_daiso.simulators import LatticeRASimulator as RASimulator
            RA_model_path = [os.path.join(ROOT_PATH, config[simulator_model][f'{i+1}F']) for i in range(len(config[simulator_model]))]
            pre_RA_model_path = [os.path.join(ROOT_PATH, config[simulator_pre_model][f'{i+1}F']) for i in range(len(config[simulator_pre_model]))]
        elif 'dynamic' in simulator_model:
            from env_daiso.simulators import DynamicRASimulator as RASimulator
            RA_model_path = None
            pre_RA_model_path = None

        HC_model_path = os.path.join(ROOT_PATH, config['simulator_HC'])
        self.simulator_RA = RASimulator(model_path=RA_model_path)
        self.simulator_pre_RA = RASimulator(model_path=pre_RA_model_path)
        self.simulator_HC = HCSimulator(model_path=HC_model_path)

        self.cost_min = 0
        self.cost_max = 438.7  # won

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state

    def get_data_size(self):
        return self.data_manager.get_data_size()
    
    def get_episode_info(self):
        episode_number = self.data_manager.get_selectedEpisodeNum()
        episode_date = self.data_manager.get_episode_date()
        return episode_number, episode_date
    
    def reset(self):
        """
        Reset episode. Reset time at DataManager.firstMinute_inEpisode (eg. 08:00:00) and return reset_state.
        """
        # Set episode index
        if "train" in self.phase: #and "pastData" in phase:
            self.data_manager.set_idx_atRandomTrainEpisode()
        elif "test" in self.phase: #and "pastData" in phase:
            self.data_manager.set_idx_atNextTestEpisode()

        # Get data row of past data
        data = self.data_manager.get_rowAsDict()  # at self.data_manager.idx

        # Set current state
        self.state = self.stateObj.state_fromData(data)

        # Initialize initial/previous control
        self.previous_action = [0, 0, 0, 0, 0]

        info = {}

        state_asArray = np.array(list(self.state.get_values()))
        return state_asArray, info

    def step(self, action):
        """
        Uses the action value to compute the next RA(return air) temperature.
        This function also computes the reward, next_state and done flag.

        Args:
            action: control taken by this environment (can be different in real world.)
        
        Returns:
            reward, next_state, done, info
        """
        timestep = self.obsObj.get_timestep_from_state(state=self.state)

        # Calculate E_ehp
        E_ehps = np.array([self.ehp * n_AC * self.unit_timestep for n_AC in action]).reshape(-1)

        # T_ra simulator model
        if self.state.minutes < self.config['DIS_OPEN_TIME'] * 60:
            delta_T_ra = self.simulator_pre_RA.simulate(self.state, E_ehps)
        else:
            delta_T_ra = self.simulator_RA.simulate(self.state, E_ehps)

        current_T_ra = np.array([self.state.T_ra_1F, self.state.T_ra_2F, self.state.T_ra_3F, self.state.T_ra_4F, self.state.T_ra_5F])
        next_T_ra = current_T_ra + delta_T_ra

        # Transition to next state
        next_timestep = timestep + 1
        minutes = self.stateObj.get_minutes_from_timestep(next_timestep)
        
        if next_timestep == self.data_manager.lastTimestep:
            done = True
            next_state = State()
            next_past_data = None
        else:
            done = False
            self.data_manager.increment_idx()
            next_past_data = self.data_manager.get_rowAsDict()

            next_state = State(
                days = self.state.days,
                minutes = minutes,
                T_ra_1F = next_T_ra[0],
                T_ra_2F = next_T_ra[1],
                T_ra_3F = next_T_ra[2],
                T_ra_4F = next_T_ra[3],
                T_ra_5F = next_T_ra[4],
                T_oa = next_past_data['T_oa'],
                T_oa_min = next_past_data['T_oa_min'],
                T_oa_max = next_past_data['T_oa_max'],
                CA = next_past_data['CA'],
                n_HC_instant = next_past_data['n_HC_instant'],
                n_HC = next_past_data['n_HC']
            )

        """ Calculate Cost and Reward """
        cop = 9.57
        cost = self.estimator.elec_cost_from_state(next_state, E_ehps, cop)
        reward, reward_terms = self.reward_function(cost, next_T_ra, self.state.minutes, action)

        self.state = next_state

        control = self.ctrlObj.control_fromAction(action)

        info = {
            'next_data': next_past_data,
            'done': done,
            'next_state': next_state,  # state
            'action': action,  # action
            'control': control,  # control
            'reward_terms': reward_terms,  # reward_terms (dictionary)
            'reward': reward,
        }

        next_state_asArray = np.array(list(next_state.get_values()))

        return next_state_asArray, reward, done, done, info

    def reward_function(self, cost, next_T_ra, minutes, action):
        """
        Each _term is a kind of cost or loss. The less the better.
        """

        # Cost Term
        cost_term = self.normalize_cost(cost)

        # Temperature Term
        _tFs = []
        for i, T_ra in enumerate(next_T_ra, start=1):
            comfort_low, comfort_high = self.comfortLows[f'{i}F'], self.comfortHighs[f'{i}F']
            gradient_t = 1
            if T_ra <= comfort_low:  # not allowed
                _tF = gradient_t * (comfort_low - T_ra)
            elif T_ra >= comfort_high:  # not allowed
                _tF = gradient_t * (T_ra - comfort_high)
            else:  # in comfort zone
                _tF = 0
            _tFs.append(_tF)
        temperature_term = sum(_tFs) / len(_tFs)

        # Consecutive constraint term
        # action and self.previous_action are np.array
        consecutive_term = np.sum((action - self.previous_action)**2)
            
        # Reward
        lambda_cost = self.config['lambda_cost']
        lambda_temperature = self.config['lambda_temperature'] if minutes >= self.config['DIS_OPEN_TIME'] * 60 else 0
        lambda_consecutive = self.config['lambda_consecutive'] if minutes > self.config['RL_START_TIME'] * 60 else 0 # init timestep 에서는 consecutive 계산 제외

        cost_term = -lambda_cost * cost_term
        temperature_term = -lambda_temperature * temperature_term
        consecutive_term = -lambda_consecutive * consecutive_term

        reward = cost_term + temperature_term + consecutive_term
        reward_dict = {
            "cost_term": cost_term,
            "temperature_term": temperature_term,
            "consecutive_term": consecutive_term
        }
        return reward, reward_dict


    def normalize_cost(self, org_cost):
        """
        Performs min-max normalization on the cost value using the legacy data.
        Args:
            org_cost (float): cost value in KRW.
        Returns:
            A min-max normalized, cost value (float).
        """
        return (org_cost - self.cost_min) / (self.cost_max - self.cost_min)


    def unnormalize_cost(self, normalized_cost):
        """
        Restore from normalized cost.
        Args:
            norm_cost (float): a min-max normalized, cost value.
        Returns:
            Cost value in KRW (float).
        """
        return normalized_cost * (self.cost_max - self.cost_min) + self.cost_min
