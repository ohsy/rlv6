from env_daiso.stateobservation_DIS import StateObj


class Estimator:
    def __init__(self, season, stateObj: StateObj):
        self.stateObj = stateObj

        self.season = season

        # 한전 2023.05.16 전기요금표 참고
        # https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00101.jsp
        # 일반용전력(갑)I - 저압
        # 다이소의 경우 시간별 전기 요금에 차등을 두지 않는 것으로 계산
        self.season_rate_data = {
            "spring": 83.9,
            "summer": 124.4,
            "fall": 83.9,
            "winter": 111.0
        }

    def get_rate(self, state: object) -> float:
        """
        get the gas and electric rate specific to the month and timestep.
        """
        elecRate = self.season_rate_data[self.season]

        #elecRate = cost_conv_rate['Elect, Won/MJ'].iloc[timestep]
        #gasRate = cost_conv_rate['Gas, Won/MJ'].iloc[timestep]
        return elecRate#, gasRate

    def elec_cost_from_state(self, state, E_ehps, cop):
        # 1 kWh = 3.6 MJ
        total_E_ehp = -sum(E_ehps) # 단위는 (60J)
        total_E_ehp *= (1 / 60)    # 단위는 (3600J) = (1Wh)
        total_E_cons = total_E_ehp / cop / 1000  # energy consumption; 단위는 kWh

        elec_rate = self.get_rate(state) # won/kWh
        cost = total_E_cons * elec_rate

        return cost

