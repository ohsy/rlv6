import os
import json

import numpy as np
import pandas as pd

from .calculateE import *
from .utils import *

with open(os.getcwd() + "/env_daiso/config.json") as json_file:
    proj_config = json.load(json_file)

if proj_config['simulator_RA_model'] == 'simple_dynamic':
    with open(os.getcwd() + "/env_daiso/dynamic_modules/simple_dynamic_simulator.json") as f:
        config = json.load(f)
else:
    with open(os.getcwd() + "/env_daiso/dynamic_modules/dynamic_simulator.json") as f:
        config = json.load(f)


def calDfs(df):
    start_date = pd.to_datetime('2023-01-01')
    df["timestamp"] = start_date + pd.to_timedelta(df['days'], unit='D') + pd.to_timedelta(df['hour'], unit='H') + pd.to_timedelta(df['minute'], unit='m')
    E_ehp_df = processEHPdata(df)
    solar_angle_df = addSolarAngle(df)
    E_solar_df = processEsolar(pd.concat([df, solar_angle_df], axis = 1), solar_const = config['solar_const'])
    E_window_df = processEwindows(df)
    E_wall_df = processEwalls(df)
    E_ceiling_df = processEceiling(df)
    E_body_df = processEbodys(df)
    E_light_df = processELights(df) # light_on: int, light_off: int
    E_infill_df = processEinfill(df)

    total_df = pd.concat([df, E_ehp_df, solar_angle_df, E_solar_df, E_window_df, E_wall_df, E_ceiling_df, E_body_df, E_light_df, E_infill_df], axis = 1)

    return total_df


def eachFloorE(merge_df):
    total_df = pd.DataFrame(0, index = range(0, len(merge_df)),
                            columns = ['E_total_1G', 'E_total_2G', 'E_total_3G', 'E_total_4G', 'E_total_5G'])
    df = merge_df[['timestamp','1F_temp','2F_temp','3F_temp','4F_temp','5F_temp']]

    for col in total_df.columns:
        floor = col[-2:]
        for c in merge_df.columns:
            if floor in c:
                total_df['E_total_' + floor] = total_df['E_total_' + floor] + merge_df[c]
        if floor == '1G':
            total_df['E_total_' + floor] = total_df['E_total_' + floor] + merge_df['Infiltration']
        elif floor == '5G':
            total_df['E_total_' + floor] = total_df['E_total_' + floor] + merge_df['Ceiling']
    total_df = pd.concat([df, total_df], axis=1).reset_index(drop=True)

    return total_df

def returnDeltaT(df):
    deltat_df = pd.DataFrame(index = range(0, len(df)), 
                            columns = ['delta_T_1G', 'delta_T_2G', 'delta_T_3G', 'delta_T_4G', 'delta_T_5G'])
    
    for col in deltat_df.columns:
        floor = col[-2:]
        deltat_df['delta_T_' + floor] = df[['E_total_'+floor, floor[0]+'F_temp']].apply(calculate_deltaDF, args=(floor[0]), axis=1)
    return deltat_df

def makeUnitTime(df, Unit=config['Unit']):
    simul_columns = []
    for i in df.columns:
        if "TA" in i:
            simul_columns.append(i)
        if "temp" in i and len(i) <= 9:
            simul_columns.append(i)
        if "int" in i:
            simul_columns.append(i)
        if "ehp" in i:
            simul_columns.append(i)
        if "window" in i:
            simul_columns.append(i)
        if "wall" in i:
            simul_columns.append(i)
        if "Ceiling" in i:
            simul_columns.append(i)
        if "body" in i:
            simul_columns.append(i)
        if "E_solar" in i:
            simul_columns.append(i)
        if "light" in i:
            simul_columns.append(i)
        if "Infiltration" in i:
            simul_columns.append(i)

    simul_columns.insert(0,"timestamp")
    simul_df = pd.DataFrame(0, index = range(0, len(df)//Unit-1), columns=simul_columns)

    for col in simul_columns:
        if ('timestamp' in col):
            simul_df[col] = df[col][::10].reset_index(drop=True)
        else:
            simul_df[col] = df[col].rolling(window=10, min_periods=1).sum().iloc[9::10].reset_index(drop=True)
            if ('temp' in col) or ('TA' in col):
                simul_df[col] = simul_df[col] / 10


    # simul_df.to_csv('data/calculate.csv', index=False)
    
    return simul_df

def coefFitting(df):
    mean_df = pd.DataFrame(columns=['timestamp','E_int','E_ehp','E_window','E_wall','E_solar','E_body','E_light','Ceiling','Infiltration','TA','temp','deltaT'])
    
    # ehp 층별 나누기
    # mean_df = pd.DataFrame(columns=['timestamp','E_int', 'E_ehp1', 'E_ehp2', 'E_ehp3', 'E_ehp4', 'E_ehp5','E_window','E_wall','E_solar','E_body','E_light','Ceiling','Infiltration','TA','temp','deltaT'])
    TA = df['TA']
    temp = (df['1F_temp']*config['vol1'] + df['2F_temp']*config['vol2'] + df['3F_temp']*config['vol3'] + df['4F_temp']*config['vol4'] + df['5F_temp']*config['vol5'])/(config['vol1'] + config['vol2'] + config['vol3'] + config['vol4'] + config['vol5'])
    deltaT = temp.diff()

    E_int = pd.DataFrame(index=range(0, len(mean_df)))
    df['time_diff'] = df['timestamp'].diff(periods=1)
    for j in range(1, 6):
        df[f'{j}F_tempdelta'] = df[f'{j}F_temp'].diff()[1:].reset_index(drop=True).append(pd.Series(0), ignore_index=True)
        df[f'E_int_{j}G'] = df[[f'{j}F_tempdelta', f'{j}F_temp']].apply(calculate_E_intDF, floor=j, axis=1)

    time = df['timestamp']
    E_int = sum_all_floors_E(df, 'int')
    E_ehp = sum_all_floors_E(df, 'ehp')
    E_window = sum_all_floors_E(df, 'window')
    E_wall = sum_all_floors_E(df, 'wall')
    E_solar = sum_all_floors_E(df, 'solar')
    E_body = sum_all_floors_E(df, 'body')
    E_light = sum_all_floors_E(df, 'light')
    Infiltration = df['Infiltration']
    ceiling = df['Ceiling']

    mean_df['timestamp'] = time
    mean_df['E_int'] = E_int
    mean_df['E_ehp'] = E_ehp
    mean_df['E_window'] = E_window
    mean_df['E_wall'] = E_wall
    mean_df['E_solar'] = E_solar
    mean_df['E_body'] = E_body
    mean_df['E_light'] = E_light
    mean_df['Ceiling'] = ceiling
    mean_df['Infiltration'] = Infiltration
    mean_df['TA'] = TA
    mean_df['temp'] = temp
    mean_df['deltaT'] = deltaT

    mean_df.to_csv('data/total_avg.csv', index=False)
