import os
import json
from datetime import datetime, timedelta

with open(os.getcwd() + "/env_daiso/config.json") as json_file:
    proj_config = json.load(json_file)

if proj_config['simulator_RA_model'] == 'simple_dynamic':
    with open(os.getcwd() + "/env_daiso/dynamic_modules/simple_dynamic_simulator.json") as f:
        config = json.load(f)
else:
    with open(os.getcwd() + "/env_daiso/dynamic_modules/dynamic_simulator.json") as f:
        config = json.load(f)


# 온도 변화에 필요한 에너지 = cp * 체적 * 온도차
def calculate_air_heat_capacity(T):
    Cp0 = 1003  # J/m³·°C
    alpha = 0.0035  # 1/°C
    Cp = Cp0 + alpha * (T - 20)  # 기준 온도 T0 = 20°C로 가정
    # Cp = 1006.32461 + 0.03491*T - 0.00109*T**2 + 1.75726*10**(-5)*T**3 - 8.37239*10**(-8)*T**4
    # 10분단위
    return Cp
def calculate_delta(E, floor, T):
    return E / calculate_air_heat_capacity(T) / config[f'vol{floor}']
def calculate_E_int(temp_diff, floor, T):
    return temp_diff * calculate_air_heat_capacity(T) * config[f'vol{floor}']
def calculate_deltaDF(x, floor):
    return x[0] / calculate_air_heat_capacity(x[1]) / config[f'vol{floor}']
def calculate_E_intDF(x, floor):
    return x[0] * calculate_air_heat_capacity(x[1]) * config[f'vol{floor}']

def to_timestamp(df):
    year = '2023'
    start_date = year + '-01-01'
    return datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=int(df['days']),
                                                                 hours=int(df['hour']),
                                                                 minutes=int(df['minute']))

def sum_all_floors_E(df, string):
    E = df['E_' + string + '_1G']
    for f in range(2, 6):
        E += df['E_' + string + f'_{f}G']
    return E

def stack_effect(E_list, temp_list, max_trans_E_ehp):
    E_potential = [0]*6
    E_transfer = [0]*6
    E_eff = [0]*6

    for j in range(5,0,-1):
        if j == 5:
            E_potential[j] = E_list[j]*max_trans_E_ehp
        else:
            E_potential[j] = (E_list[j] + E_transfer[j+1])*max_trans_E_ehp
        if j == 1:
            E_transfer[j] = 0
        else:
            E_transfer[j] = E_potential[j]*max((temp_list[j]-temp_list[j-1])/5.5,0)
        if j == 5:
            E_eff[j] = E_list[j] - E_transfer[j]
        else:
            E_eff[j] = E_list[j] + E_transfer[j+1] - E_transfer[j]
    
    return E_eff
