import os, sys
import random
from collections import deque
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#from .bcolor import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataManager:
    """
    Times are in unit of minute since hour cannot deal with vaious timesteps, eg. 10-minute timestep.
    For example, when building opens 10:30 AM, openingMinute = 10 * 60 + 30.
    """

    def __init__(self, config, phase, current_time=None):
        self.config = config
        self.current_time = current_time

        data_path = "./env_daiso/data.csv"

        self.unit_timestep = self.config["unit_timestep"]
        self.firstMinute_inEpisode = self.config["RL_START_TIME"] * 60  # timesteps before this are not considered in episode
        self.lastMinute_inEpisode = self.config["RL_END_TIME"] * 60  # timesteps after this are not considered in episode

        self.firstTimestep = 0
        self.lastTimestep = (self.lastMinute_inEpisode - self.firstMinute_inEpisode) // self.unit_timestep
        self.minutes_inEpisode = self.lastMinute_inEpisode - self.firstMinute_inEpisode
        self.nTimesteps_perEpisode = self.minutes_inEpisode // self.unit_timestep

        if phase in ['train', 'test', 'continued_train']:
            self.data = self._load_data(data_path)
            self.idx = 0  # for self.data

            episodes = np.unique(self.data['date'].to_numpy())
            episode_numbers = np.arange(len(episodes))
            self.trainEpisodeNums, testEpisodeNums = train_test_split(episode_numbers, test_size=config["TEST_SIZE"], shuffle=True)
            self.testEpisodeNums = sorted(testEpisodeNums)
            self.testEpisodeCnt = 0

            print(self.data)
            print(f'[Data INFO] #episodes(dates): {len(episodes)} (#train: {len(self.trainEpisodeNums)}, #test: {len(self.testEpisodeNums)})')
            print()

    def _load_data(self, data_path):
        # Load RL train data
        raw_data = pd.read_csv(data_path).dropna()

        # Sampling data based on the unit time step
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], format="%Y-%m-%d %H:%M:%S")
        raw_data = raw_data.set_index(raw_data['timestamp'])
        data = raw_data.resample(f"{self.config['unit_timestep']}T").mean(numeric_only=True).reset_index()
        
        n_HC_instant = raw_data['n_HC_instant'].resample(f"{self.config['unit_timestep']}T").sum(numeric_only=True).reset_index()
        data['n_HC_instant'] = n_HC_instant['n_HC_instant'] # 1분 인원은 unit_timestep 동안의 sum

        E_ehp_cols = ['E_ehp_1', 'E_ehp_2', 'E_ehp_3', 'E_ehp_4', 'E_ehp_5']
        E_ehp = raw_data[E_ehp_cols].resample(f"{self.config['unit_timestep']}T").sum(numeric_only=True).reset_index()
        data[E_ehp_cols] = E_ehp[E_ehp_cols] # E_ehp는 unit_timestep 동안의 sum

        print(f"Unit Timestep : {self.config['unit_timestep']} min")

        _timestamp = data['timestamp']
        data.insert(1, "minutes", _timestamp.dt.hour*60 + _timestamp.dt.minute)
        days = (_timestamp - pd.to_datetime(f'2023-01-01')).dt.days
        data.insert(1, "days", days)
        data.insert(1, "date", _timestamp.dt.date)
        data = data.dropna() # date 추가할 때 비어있는 날짜도 포함되는 문제가 있음.
        
        timestep_fromData = (_timestamp.iloc[1] - _timestamp.iloc[0]).seconds // 60
        assert self.unit_timestep == timestep_fromData, "timestep={} != timestep_fromData={}".format(self.unit_timestep, timestep_fromData)

        firstTime_inEpisode = time(hour=self.firstMinute_inEpisode // 60, minute=self.firstMinute_inEpisode % 60)
        lastTime_inEpisode = time(hour=self.lastMinute_inEpisode // 60, minute=self.lastMinute_inEpisode % 60)
        data = data[(_timestamp.dt.time >= firstTime_inEpisode) & (_timestamp.dt.time < lastTime_inEpisode)].reset_index(drop=True)

        # Determine Outside Temperature min/Max
        T_oa_max = []  # max T_oa in [current timestep : last timestep in current episode]
        T_oa_min = []  # min T_oa in [current timestep : last timestep in current episode]
        for _idx in range(len(data)):
            _end_idx = (_idx // self.nTimesteps_perEpisode + 1) * self.nTimesteps_perEpisode  # index of final row for current episode
            T_oa_max.append(data['T_oa'].iloc[_idx:_end_idx].max())
            T_oa_min.append(data['T_oa'].iloc[_idx:_end_idx].min())
        data.loc[:, 'T_oa_max'] = T_oa_max
        data.loc[:, 'T_oa_min'] = T_oa_min

        assert len(data) % self.nTimesteps_perEpisode == 0, \
                f"#data={len(data)} is not a multiple of nTimesteps_perEpisode={self.nTimesteps_perEpisode}"

        return data

    def get_data_size(self):
        return len(self.trainEpisodeNums), len(self.testEpisodeNums)

    def get_idx(self):
        return self.idx

    def set_idx(self, idx=None, phase='deploy'):
        if phase == "deploy":
            timestamp = self.current_time
            hh = timestamp.hour
            mm = timestamp.minute
            current_timestep = (hh * 60 + mm) // self.unit_timestep
            self.idx = current_timestep
        else:
            self.idx = idx

    def increment_idx(self):
        self.idx += 1

    def set_idx_atRandomTrainEpisode(self):
        self.selectedEpisodeNum = random.choice(self.trainEpisodeNums)
        self.idx = self.selectedEpisodeNum * self.nTimesteps_perEpisode

    def set_idx_atNextTestEpisode(self):
        self.selectedEpisodeNum = self.testEpisodeNums[self.testEpisodeCnt]
        self.idx = self.selectedEpisodeNum * self.nTimesteps_perEpisode
        self.testEpisodeCnt += 1

    def get_selectedEpisodeNum(self):
        return self.selectedEpisodeNum

    def get_rowAsDict(self, idx_offset=0):
        assert 0 <= self.idx + idx_offset < len(self.data), f"index {self.idx}+{idx_offset} is out of bounds where dataLen={len(self.data)}"
        return self.data.iloc[self.idx + idx_offset].to_dict()

    def get(self, column, default=0, idx_offset=0):
        """
        Args:
            default: return default if index (== self.idx + idx_offset) is out of current episode
        """
        if idx_offset == 0:
            value = self.data[column].iloc[self.idx]
        else:
            low = (self.idx // self.nTimesteps_perEpisode) * self.nTimesteps_perEpisode
            high = (self.idx // self.nTimesteps_perEpisode + 1) * self.nTimesteps_perEpisode
            if low <= self.idx + idx_offset < high:
                value = self.data[column].iloc[self.idx + idx_offset]
            else:
                value = default
        return value
    
    def get_episode_date(self):
        return self.data['date'].iloc[self.idx]

    def get_current_time(self):
        return self.current_time
   
    def isOpen(self):
        timestamp = self.get('timestamp', idx_offset=0)
        koreanMinute = 60 * timestamp.hour + timestamp.minute
        return self.openingMinute <= koreanMinute <= self.closingMinute

    def get_rate(self, month, timestep):
        """
        get the gas and electric rate specific to the month and timestep.
        """
        cost_conv_rate = self.season_rate_data[self.season_forMonth[month]]
        # print("cost_conv_rate=\n{}".format(cost_conv_rate))
        # print("get_rate: month={}, timestep={}".format(month, timestep))
        # print("get_rate: type(month)={}, type(timestep)={}".format(type(month), type(timestep)))
        elecRate = cost_conv_rate[self.col_elecRate].iloc[timestep]
        gasRate = cost_conv_rate[self.col_gasRate].iloc[timestep]
        # elecRate = 0
        # gasRate = 0
        return elecRate, gasRate

    def normalize_energy(self, original):
        """
        TEMP: eProd vs. eUsed_ahu
        positive or negative sign is not changed.
        """
        if original > 0:
            normalized = original / self.eUsed_ahu_max
        elif original < 0:
            normalized = -(original / self.eUsed_ahu_min)
        else:
            normalized = 0
        return normalized

    def unnormalize_energy(self, normalized):  # TEMP: eProd vs eUsed_ahu
        """
        restore from normalized energy
        """
        if normalized > 0:
            # unnormalized = normalized * self.eUsed_ahu_max * self.relaxA
            unnormalized = normalized * self.eUsed_ahu_max  # TEMP
        elif normalized < 0:
            # unnormalized = -(normalized * self.eUsed_ahu_min * self.relaxA)
            unnormalized = -(normalized * self.eUsed_ahu_min)  # TEMP
        else:
            unnormalized = 0
        return unnormalized


if __name__ == "__main__":
    import os, sys
    from config import get_config
    config = get_config()

    current_time = datetime.now()
    current_time -= timedelta(minutes=(current_time.minute % config["unit_timestep"]))

    data_manager = DataManager(config, season='summer', current_time=current_time)

