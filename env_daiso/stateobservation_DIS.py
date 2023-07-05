from datetime import datetime, timedelta

from .base_type import *


@dataclass
class State(BaseTypeClass):
    days: int = 0 # [0, 364] / 01-01 == 0
    minutes: int = 0 # = hour * 60 + minute
    T_ra_1F: float = 0. # Temperature of room air
    T_ra_2F: float = 0.
    T_ra_3F: float = 0.
    T_ra_4F: float = 0.
    T_ra_5F: float = 0.
    T_oa: float = 0. # T of outside air
    T_oa_min: float = 0. # min T_oa from this timestep to end
    T_oa_max: float = 0. # max T_oa from this timestep to end
    CA: float = 0. # clouds
    n_HC_instant: float = 0. # head count of that time
    n_HC: float = 0. # cumulative head count


class StateObj:
    def __init__(self, config: dict):
        self.config = config
        
        state = State()
        self.n_elements = len(state)  # num of elements

        self.start_time = config['RL_START_TIME']
        self.unit_timestep = config['unit_timestep']

    def get_instance(self):
        return State()

    def get_timestep_from_state(self, state: State):
        timestep = state.minutes - (60 * self.start_time)
        timestep = timestep // self.unit_timestep
        return timestep
    
    def get_minutes_from_timestep(self, timestep: int):
        _t = timestep * self.unit_timestep
        hour = _t // 60 + self.start_time
        minute = _t % 60
        return hour*60 + minute

    def get_datetime(self, state: State):
        dt = datetime.strptime("2023-01-01 00:00", "%Y-%m-%d %H:%M")
        hour = state.minutes // 60
        minute = state.minutes - (state.minutes // 60) * 60
        dt += (timedelta(days=state.days) + timedelta(hours=hour) + timedelta(minutes=minute))
        return dt

    def state_fromData(self, data: dict) -> State:
        """
        Args:
            data (dict): returned from data_manager
        """
        state = State( 
            days = data['days'],
            minutes = data['minutes'],
            T_ra_1F = data['T_ra_1F'],
            T_ra_2F = data['T_ra_2F'],
            T_ra_3F = data['T_ra_3F'],
            T_ra_4F = data['T_ra_4F'],
            T_ra_5F = data['T_ra_5F'],
            T_oa = data['T_oa'],
            T_oa_min = data['T_oa_min'],
            T_oa_max = data['T_oa_max'],
            CA = data['CA'],
            n_HC = data['n_HC'],
            n_HC_instant = data['n_HC_instant']
        )
        return state


@dataclass
class Observation(BaseTypeClass):
    days: int = 0
    timestep: int = 0 # 0 at open time, increase every unit_timestep(minutes)
    T_ra_1F_by_comfort_low: float = 0. # == T_ra_1F - comfortLow
    T_ra_2F_by_comfort_low: float = 0.
    T_ra_3F_by_comfort_low: float = 0.
    T_ra_4F_by_comfort_low: float = 0.
    T_ra_5F_by_comfort_low: float = 0.
    T_ra_1F_by_comfort_high: float = 0. # == T_ra_1F - comfortHigh
    T_ra_2F_by_comfort_high: float = 0.
    T_ra_3F_by_comfort_high: float = 0.
    T_ra_4F_by_comfort_high: float = 0.
    T_ra_5F_by_comfort_high: float = 0.
    T_ra_1F_by_T_oa: float = 0. # == T_ra_1F - T_oa
    T_ra_2F_by_T_oa: float = 0.
    T_ra_3F_by_T_oa: float = 0.
    T_ra_4F_by_T_oa: float = 0.
    T_ra_5F_by_T_oa: float = 0.
    T_oa_min: float = 0. # min T_oa from this timestep to end
    T_oa_max: float = 0. # max T_oa from this timestep to end
    CA: float = 0.
    n_HC_instant: float = 0.
    n_HC: float = 0.


class ObservationObj:
    def __init__(self, config: dict):
        self.config = config

        observation = Observation()
        self.n_elements = len(observation)
        
        self.start_time = config['RL_START_TIME']
        self.unit_timestep = config['unit_timestep']

        self._set_comfortZone()

    def get_instance(self):
        return Observation()

    def get_observation_size(self):
        return self.n_elements
    
    def _set_comfortZone(self):
        agentMode = self.config["AgentMode"]
        self.comfortLows, self.comfortHighs = {}, {}
        for floor, value in self.config["COMFORT_ZONE"][agentMode].items():
            low, high = value
            self.comfortLows[floor] = low
            self.comfortHighs[floor] = high

    def get_comfortZone(self):
        return self.comfortLows, self.comfortHighs
    
    def isinComfortZone(self, state: State):
        T_ras = [state.T_ra_1F, state.T_ra_2F, state.T_ra_3F, state.T_ra_4F, state.T_ra_5F]

        isinComfortZone = []
        for i, T_ra in enumerate(T_ras):
            isin = (T_ra >= self.comfortLows[f'{i+1}F']) & (T_ra <= self.comfortHighs[f'{i+1}F'])
            isinComfortZone.append(int(isin))
        return isinComfortZone
    
    def get_timestep_from_state(self, state: State):
        timestep = state.minutes - (60 * self.start_time)
        timestep = timestep // self.unit_timestep
        return timestep
    
    def hour_minute_from_timestep(self, timestep):
        timestep *= self.unit_timestep
        hour = timestep // 60 + self.start_time
        minute = timestep % 60
        return hour, minute
    
    def get_datetime(self, obs: Observation):
        dt = datetime.strptime("23-01-01 00:00", "%y-%m-%d %H:%M")
        hour, minute = self.hour_minute_from_timestep(obs.timestep)
        dt += (timedelta(days=obs.days) + timedelta(hours=hour) + timedelta(minutes=minute))
        return dt

    def observation_fromState(self, state: State) -> Observation:
        observation = Observation(
            days = state.days,
            timestep = self.get_timestep_from_state(state=state),
            T_ra_1F_by_comfort_low = state.T_ra_1F - self.comfortLows['1F'],
            T_ra_2F_by_comfort_low = state.T_ra_2F - self.comfortLows['2F'],
            T_ra_3F_by_comfort_low = state.T_ra_3F - self.comfortLows['3F'],
            T_ra_4F_by_comfort_low = state.T_ra_4F - self.comfortLows['4F'],
            T_ra_5F_by_comfort_low = state.T_ra_5F - self.comfortLows['5F'],
            T_ra_1F_by_comfort_high = state.T_ra_1F - self.comfortHighs['1F'],
            T_ra_2F_by_comfort_high = state.T_ra_2F - self.comfortHighs['2F'],
            T_ra_3F_by_comfort_high = state.T_ra_3F - self.comfortHighs['3F'],
            T_ra_4F_by_comfort_high = state.T_ra_4F - self.comfortHighs['4F'],
            T_ra_5F_by_comfort_high = state.T_ra_5F - self.comfortHighs['5F'],
            T_ra_1F_by_T_oa = state.T_ra_1F - state.T_oa,
            T_ra_2F_by_T_oa = state.T_ra_2F - state.T_oa,
            T_ra_3F_by_T_oa = state.T_ra_3F - state.T_oa,
            T_ra_4F_by_T_oa = state.T_ra_4F - state.T_oa,
            T_ra_5F_by_T_oa = state.T_ra_5F - state.T_oa,
            T_oa_min = state.T_oa_min,
            T_oa_max = state.T_oa_max,
            CA = state.CA,
            n_HC_instant = state.n_HC_instant,
            n_HC = state.n_HC,
        )
        return observation
