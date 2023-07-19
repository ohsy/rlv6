import os
import json
import math

import numpy as np
import pandas as pd

from .utils import *


with open(os.getcwd() + "/env_daiso/config.json") as json_file:
    proj_config = json.load(json_file)

if proj_config['simulator_RA_model'] == 'simple_dynamic':
    with open(os.getcwd() + "/env_daiso/dynamic_modules/simple_dynamic_simulator.json") as f:
        config = json.load(f)
else:
    with open(os.getcwd() + "/env_daiso/dynamic_modules/dynamic_simulator.json") as f:
        config = json.load(f)
        print('simple')


def processEHPdata(merged_df, COP=config['COP'], max_trans_E_ehp=config['max_trans_E_ehp']) : 
    E_ehp_df = pd.DataFrame(index = range(0, len(merged_df)), 
                            columns = ['E_ehp_1G', 'E_ehp_2G', 'E_ehp_3G', 'E_ehp_4G', 'E_ehp_5G'])

    for i in E_ehp_df.columns:
        num = i[-2]
        E_ehp_df[i] = merged_df['demand' + num]

    # # stack effect
    # selected_columns = [col for col in merged_df.columns if 'temp' in col]
    # temp_df = merged_df[selected_columns]
    # for i in range(len(E_ehp_df)):
    #     ehp_list = E_ehp_df.loc[i].values.tolist()
    #     temp_list = temp_df.loc[i].values.tolist()
    #     E_eff = stack_effect(ehp_list, temp_list, max_trans_E_ehp)
    #     E_ehp_df.loc[i] = E_eff

    return E_ehp_df

# days, B, Offset, declination angle, solar hour angle, solar elevation angle 
def addSolarAngle(merge_df, latitude=config['latitude'], standard_longitude=config['standard_longitude'], longitude=config['longitude']) :

    solar_df = pd.DataFrame(index = range(0, len(merge_df)), 
                            columns = ['days', 'B', 'Offset', 'declination_angle(degree)', 'solar_hour_angle(degree)', 'solar_elevation_angle(degree)', 'solar_time'])
    solar_df['days'] = merge_df['days']
    # B
    solar_df['B'] = (solar_df['days'] - 1) * 2 * math.pi / 365
    # Offset
    solar_df['Offset']= 4 * (standard_longitude - longitude)\
                        + 229.2 * (0.000075 + 0.001868 * solar_df['B'].apply(math.cos)
                                            - 0.032077 * solar_df['B'].apply(math.sin)
                                            - 0.014615 * (2 * solar_df['B']).apply(math.cos)
                                            - 0.040890 * (2 * solar_df['B']).apply(math.sin))
    # solar_time
    solar_df['solar_time'] = merge_df['timestamp'] + pd.to_timedelta(solar_df['Offset'], unit='m')
    # declination angle(degree)
    solar_df['declination_angle(degree)'] = -23.44 * (2 * math.pi * (solar_df['days'] + (solar_df['solar_time'].dt.hour + solar_df['solar_time'].dt.minute / 60)/ 24 + 10) / 365).apply(math.cos)

    # solar hour angle(degree)
    solar_df['solar_hour_angle(degree)'] = (solar_df['solar_time'].dt.hour + solar_df['solar_time'].dt.minute / 60 - 12) * 15

    solar_df['solar_elevation_angle(degree)'] = 180 / math.pi \
                                                * (math.sin(latitude * math.pi / 180) * (solar_df['declination_angle(degree)'] * math.pi / 180).apply(math.sin) \
                                                + math.cos(latitude * math.pi / 180) * (solar_df['declination_angle(degree)'] * math.pi / 180).apply(math.cos) \
                                                * (solar_df['solar_hour_angle(degree)'] * math.pi / 180).apply(math.cos)).apply(math.asin)
    
    return solar_df

def processEsolar(merge_df, solar_const: int, window_angle_degree=config['window_angle_degree'], baseline=config['Base_Line'], transmittancy=config['transmittancy']):

    # E_solar       
    E_solar_df = pd.DataFrame(index = range(0, len(merge_df)), 
                                columns = ['E_solar_1G', 'E_solar_2G', 'E_solar_3G', 'E_solar_4G', 'E_solar_5G'])
    
    Constant_dict = {
            '1' : [1230, 0.142, 0.058],
            '2' : [1215, 0.144, 0.060],
            '3' : [1186, 0.156, 0.071],
            '4' : [1136, 0.180, 0.097],
            '5' : [1104, 0.196, 0.121],
            '6' : [1088, 0.205, 0.134],
            '7' : [1094, 0.186, 0.138],
            '8' : [1107, 0.201, 0.122],
            '9' : [1151, 0.177, 0.092],
            '10' : [1192, 0.160, 0.073],
            '11' : [1221, 0.149, 0.063],
            '12' : [1233, 0.142, 0.057]
        }
    merge_df['month'] = merge_df['timestamp'].apply(lambda x: x.month)

    # 태양 입사각
    solar_hour_angle_rad = merge_df['solar_hour_angle(degree)'] * math.pi / 180
    solar_elevation_angle_rad = merge_df['solar_elevation_angle(degree)'] * math.pi / 180
    solar_cos_theta = solar_hour_angle_rad.apply(math.cos) * solar_elevation_angle_rad.apply(math.cos) * math.sin(window_angle_degree * math.pi / 180) \
                    + solar_hour_angle_rad.apply(math.sin) * math.cos(window_angle_degree * math.pi / 180)

    # 수평면 확산일사에 대한 수직면 확산일사의 비
    Y = np.where(solar_cos_theta > -0.2, 0.55 + 0.437 * solar_cos_theta + 0.313 * solar_cos_theta ** 2, 0.45)

    month = merge_df['month'].astype(str)
    C = np.array([Constant_dict[m][2] for m in month])
    A = np.array([Constant_dict[m][0] for m in month])
    B = np.array([Constant_dict[m][1] for m in month])

    # 맑은 날의 법선 직달 일사량
    condition = solar_hour_angle_rad.apply(math.sin)
    solar_direct_normal =  np.where(condition > 0,
                                (A / (B / solar_hour_angle_rad.apply(math.sin)).apply(math.exp)) * (baseline + (10 - merge_df['CA']) / 10 * (1-baseline)),
                                0)

    # 일사량 (단위면적 당)
    e_solar_direct = solar_direct_normal * solar_cos_theta
    e_solar_diff = C * Y * solar_direct_normal
    e_solar_reflect = solar_direct_normal * (C + solar_hour_angle_rad.apply(math.sin)) * 0.2 * (1 - np.cos(window_angle_degree * math.pi / 180)) / 2
    e_solar = e_solar_direct + e_solar_diff + e_solar_reflect
    for col in E_solar_df.columns :
        floor = col[-2] + 'F'
        if floor == '1F':
            e_solar_value = e_solar * config['south_window'][floor] * config['transmittancy_1F']
        else:    
            e_solar_value = e_solar * config['south_window'][floor] * config['transmittancy']


        # e_solar_value = solar_const * (base_line + (10 - merge_df['CA'][i]) / 10 * (1-base_line)) \
        #               * back_data[floor]['south_window_area'] * config['transmittancy'] * solar_cos_theta

        E_solar_df[col] = np.where(e_solar_value > 0, e_solar_value, 0)       
    
    return E_solar_df


def processEints(merge_df) :

    E_int_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['E_int_1G', 'E_int_2G', 'E_int_3G', 'E_int_4G', 'E_int__5G'])

    for col in E_int_df.columns :
        floor = col[-2] + 'F'
        merge_df['temp_diff'] = merge_df[floor + '_temp'].diff()
        merge_df.loc[0, 'temp_diff'] = 0
        E_int_df[col] = merge_df[['temp_diff', f'{floor}_temp']].apply(calculate_E_intDF, floor=col[-2], axis=1)

    return E_int_df

def processEwindows(merge_df, TempDiff=config['TempDiff']) :

    E_window_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['E_window_1G', 'E_window_2G', 'E_window_3G', 'E_window_4G', 'E_window_5G'])

    for col in E_window_df.columns :
        floor = col[-2] + 'F'
        E_window_df[col] = (merge_df['TA'] + TempDiff - merge_df[floor + '_temp']) \
                            * (config['total_window'][floor] * config['U_value']['glass']) * 60

    return E_window_df

def processEwalls(merge_df, TempDiff=config['TempDiff']) :

    E_wall_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['E_wall_1G', 'E_wall_2G', 'E_wall_3G', 'E_wall_4G', 'E_wall_5G'])
    
    # ($NC6+$QP$18-OP6)*QO$10*$QP$17
    # (TA + TempDiff - ?층 평균값) * ?층 전체 창 면적 * h_eff 
    for col in E_wall_df.columns :

        floor = col[-2] + 'F'
        E_wall_df[col] = (merge_df['TA'] + TempDiff - merge_df[floor + '_temp']) \
                                * (config['total_wall'][floor] * config['U_value']['wall']) * 60
                        
    return E_wall_df

def processEceiling(merge_df, TempDiff=config['TempDiff']) :

    E_ceiling_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['Ceiling'])
    
    # ($NC6+$QP$18-OP6)*QO$10*$QP$17
    # (TA + TempDiff - ?층 평균값) * ?층 전체 창 면적 * h_eff 
    E_ceiling_df['Ceiling'] = (merge_df['TA'] + TempDiff - merge_df['5F_temp']) \
                        * config['total_wall']['ceiling'] * config['U_value']['ceiling'] * 60
        
    return E_ceiling_df

def processEbodys(merge_df, Q_h=config['Q_h'], ratio=config['human_ratio']) :

    E_body_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['E_body_1G', 'E_body_2G', 'E_body_3G', 'E_body_4G', 'E_body_5G'])
    
    for col in E_body_df.columns :
        floor = col[-2] + 'F'
        E_body_df[col] = merge_df['cumul_head'] * Q_h * ratio[floor]

    return E_body_df

def processELights(merge_df, Q_i=config['Q_l']) :

    E_light_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['E_light_1G', 'E_light_2G', 'E_light_3G', 'E_light_4G', 'E_light_5G'])
        
    # IF(OR($Q4+$R4/60<$QP$23, $Q4+$R4/60>$QP$24), 0, QO$4*$QP$22)
    # Hour + min / 60 < light_on 이거나 Hour + min / 60 > light_off 라면
    # 0 
    # 그렇지 않으면 ?층 면적 * Q_i
    for col in E_light_df.columns :
        floor = col[-2] + 'F'
        E_light_df[col] = config['S'][floor] * Q_i 

    return E_light_df

            
def processEinfill(merge_df, TempDiff=config['TempDiff'], one_infill=config['one_infill']) :

    E_infill_df = pd.DataFrame(index = range(0, len(merge_df)), 
                             columns = ['Infiltration'])
    
    temp = (merge_df['TA'] + TempDiff - merge_df['1F_temp'])

    if proj_config['simulator_RA_model'] == 'simple_dynamic':
        E_infill_df['Infiltration'] = temp * merge_df['instant_head'] * one_infill
    else:
        E_infill_df['Infiltration'] = np.where(temp > 5, temp * merge_df['instant_head'] * one_infill,
                                            temp * merge_df['instant_head'] * one_infill * (temp / 5)**2)
    

    return E_infill_df
