import os
from pathlib import Path

path_cwd = '/mnt/workspace/wangding/Desktop/TransWorldNG/transworld'

os.chdir(Path(path_cwd).absolute())
os.getcwd()

import matplotlib.pyplot as plt
import networkx as nx
import imageio
from pathlib import Path
from graph.load import load_graph
from eval.eval import *
from graph.process import generate_unique_node_id
import pandas as pd
import os
#import cv2
import pickle
import torch
from matplotlib.pyplot import figure
import math
import numpy as np
plt.style.use('default')
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import re

import glob
import pickle
from collections import defaultdict
from math import sqrt

import seaborn as sns

sns.set(style='whitegrid', palette='dark')

def idm_acceleration(v, v_lead, d_lead, a_max, b, delta, s0, T):
    """
    Computes the acceleration according to the Intelligent Driver Model (IDM).

    Args:
        v (float): The current speed of the vehicle.
        v_lead (float): The speed of the lead vehicle.
        d_lead (float): The distance between the lead vehicle and the vehicle
            being modelled.
        a_max (float): The maximum acceleration.
        b (float): The deceleration term.
        delta (float): The desired time headway.
        s0 (float): The minimum distance to the leader.
        T (float): The driver's desired time to reach their desired speed.

    Returns:
        float: The acceleration of the vehicle.
    """
    s_star = s0 + max(0, v*delta + (v*v - v_lead*v_lead)/(2*np.sqrt(a_max*b)))
    a = a_max * (1 - (v/v_lead)**delta - (s_star/d_lead)**2)
    
    return a


# def gipps_acceleration(v, v_lead, d_lead, a_max, b, v_des, tau, s_0):

#     # calculating acceleration based on Gipps'
#     acc = a_max * (1 - (v/v_des)**4 - ((s_0 + v*tau)/(d-v*tau))**2)

#     return acc

def get_feat_list_df(feat_dict, node_type, node_id, feat_list):
    df_list = []
    for feat in feat_list:
        if feat in feat_dict[node_type][node_id]:
            feat_lst = feat_dict[node_type][node_id][feat].tolist()
            time_lst = feat_dict[node_type][node_id]['time'].tolist()
            g_feat = pd.DataFrame(np.array([feat_lst, time_lst]).T)
            g_feat.columns = [feat+str(node_id), 'time']
            g_feat['time'] = g_feat['time'].astype(int)
            df_list.append(g_feat)
        else:
            print(f"{feat} column not found in feature dictionary for node {node_id}.")
    if len(df_list) > 0:
        merged_df = df_list[0]
        for df in df_list[1:]:
            merged_df = pd.merge(merged_df, df, on='time')
        return merged_df
    else:
        return pd.DataFrame(columns=['time'])

def df_cfm(feat_dict,front_veh_id, behind_veh_id):
    feat_list = ['speed','acceleration', 'pos_on_lane']
    front_feats = get_feat_list_df(feat_dict, 'veh', front_veh_id, feat_list)
    behind_feats = get_feat_list_df(feat_dict, 'veh', behind_veh_id, feat_list)
    merged = pd.merge(front_feats,behind_feats, on='time', how='inner')
    merged["distance"] = merged['pos_on_lane'+ str(front_veh_id)] - merged['pos_on_lane'+ str(behind_veh_id)]
    return merged

def CF_plot(real_struc, real_feat_dict,sim_feat_dict,front_veh_id,behind_veh_id,feat_name,plot_cf):
    cf_struc = get_cf_df(real_struc)
    veh_pair = cf_struc[(cf_struc['front_veh'] == front_veh_id) & (cf_struc['behind_veh'] == behind_veh_id)]

    feat_front = get_feat_df(real_feat_dict, 'veh', front_veh_id, feat_name)
    feat_behind = get_feat_df(real_feat_dict, 'veh', behind_veh_id, feat_name)
    feat_front_sim = get_feat_df(sim_feat_dict, 'veh', front_veh_id, feat_name)
    feat_behind_sim = get_feat_df(sim_feat_dict, 'veh', behind_veh_id,feat_name)

    merge_front = pd.merge(veh_pair, feat_front, on='time')
    merge_behind = pd.merge(merge_front, feat_behind, on='time')
    merge_front_sim = pd.merge(veh_pair, feat_front_sim, on='time')
    merge_behind_sim = pd.merge(merge_front_sim, feat_behind_sim, on='time')
    
    if plot_cf == 1:
        plt.plot(merge_behind['time'], merge_behind[feat_name+str(front_veh_id)], label = "Front_car")
        plt.plot(merge_behind['time'], merge_behind[feat_name+str(behind_veh_id)], label = "Follower_car")
        plt.plot(merge_behind_sim['time'], merge_behind_sim[feat_name+str(front_veh_id)], linestyle='--',  marker='o', markersize=2, label = "Front_car_sim")
        plt.plot(merge_behind_sim['time'],merge_behind_sim[feat_name+str(behind_veh_id)], linestyle='--',  marker='o',
            markerfacecolor='blue', markersize=2, label = "Follower_car_sim")
        plt.xlabel("Time step")
        plt.ylabel(feat_name.capitalize())
        plt.title(feat_name.capitalize() + " of vehicle pairs")
        plt.legend(fontsize=10)
        plt.show()
    else:
        return merge_behind_sim

exp_dir = Path(path_cwd).resolve().parent / "experiment"

training_step = 50
pred_step = 10
prev_step = 20

#traci_tls
# real_data_dir = exp_dir / "hangzhou" / "data" / "run1" / "test_data"
# train_data_dir = exp_dir / "hangzhou" / "data" / "run1" / "train_data"
# out_dir = exp_dir / "hangzhou" / "data" / "run1" / "out_dim_50_n_heads_4_n_layer_4_pred_step_10"



# hd_data_dir = exp_dir / "HighD" / "highway01" / "02_tracks.csv" 
real_data_dir = exp_dir / "HighD" / "highway01" / "data"
train_data_dir = exp_dir / "HighD" / "highway01" / "data"
out_dir =  exp_dir / "HighD" / "highway01" / "FineTuneModel" / "out_dim_100_n_heads_4_n_layer_4_pred_step_10"

# real_data = pd.read_csv(hd_data_dir)
# real_data['frame'] = real_data['frame'].astype(float)
# real_data = real_data[real_data['frame'] > 0]
# real_data = real_data[real_data['frame'] < 1100]
# real_feat = real_data[real_feat_name]


sim_feat = load_model_result(train_data_dir,out_dir,training_step,pred_step)

node_all = pd.read_csv(train_data_dir / "node_all.csv")
node_id_dict = generate_unique_node_id(node_all)

real_struc, real_feat, node_id_dict, scalers =  load_graph(real_data_dir,training_step-prev_step,training_step+pred_step*10,node_id_dict)

sim_feat = unscale_feat(sim_feat, scalers)
real_feat = unscale_feat(real_feat, scalers)

cf_struc = get_cf_df(real_struc)
df = cf_struc[['front_veh','behind_veh']].drop_duplicates()
all_veh_pairs = list(set([(i,j) for i in df['front_veh'] for j in df['behind_veh']]))

long_cf_pairs = []
for pair in all_veh_pairs:
    front_veh_id, behind_veh_id = pair
    veh_pair = cf_struc[(cf_struc['front_veh'] == front_veh_id) & (cf_struc['behind_veh'] == behind_veh_id)]
    pair_time = len(set(veh_pair.time))
    if pair_time>50: 
        # print(min(veh_pair['time']))
        if max(veh_pair['time'])>0:
            long_cf_pairs.append((front_veh_id, behind_veh_id, pair_time))

#for front_veh_id, behind_veh_id, pair_time in long_cf_pairs:
front_veh_id, behind_veh_id, pair_time = long_cf_pairs[0]
print(front_veh_id, behind_veh_id, pair_time)
real_merged = df_cfm(real_feat,front_veh_id, behind_veh_id)
sim_merged =  df_cfm(sim_feat,front_veh_id, behind_veh_id)

# #for front_veh_id, behind_veh_id, pair_time in long_cf_pairs:
# front_veh_id, behind_veh_id, pair_time = 72,73,67
# print(front_veh_id, behind_veh_id, pair_time)
# real_merged = df_cfm(real_feat,front_veh_id, behind_veh_id)
# sim_merged =  df_cfm(sim_feat,front_veh_id, behind_veh_id)


# extract the set of unique values in the 'step' column of each dataframe
set1 = set(real_merged['time'])
set2 = set(sim_merged['time'])

# find the intersection of the two sets of values
common_set = set1.intersection(set2)

# filter each dataframe based on the common set of values
real_merged = real_merged[real_merged['time'].isin(common_set)]
sim_merged = sim_merged[sim_merged['time'].isin(common_set)]

# Compute the IDM model output for the front vehicle
a_real = real_merged['acceleration'+ str(behind_veh_id )].values 
v_real = real_merged['speed'+ str(behind_veh_id )].values 
v_lead = real_merged['speed'+ str(front_veh_id)].values 
d_lead = real_merged['distance'].values

v_transworld = sim_merged['speed'+ str(behind_veh_id )].values 
a_transworld = sim_merged['acceleration'+ str(behind_veh_id )].values 

a_max = 5
b = 2
delta_t = 1
d0 = 2
s0 = 2
T = 1.5

v_idm = [v_real[0]]   # start with the initial speed
for i in range(1, len(v_real)):
    idm_acc = idm_acceleration(v_idm[i-1], v_lead[i-1], d_lead[i-1], a_max, b, delta_t, s0, T)
    #gipps_acc = gipps_acceleration(v_idm[i-1], v_lead[i-1], d_lead[i-1], a_max, b, s0)

    idm_speed = max(0, v_idm[i-1] + idm_acc * delta_t)
    #gipps_speed = max(0, v_idm[i-1] + gipps_acc * delta_t)

    v_idm.append(idm_speed)
    #v_gipps.append(gipps_speed)

a_idm = np.gradient(v_idm, delta_t)
#a_gipps = np.gradient(v_gipps, delta_t)

# Plot the IDM model output and real values for the front vehicle
time = real_merged['time'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))

ax1.plot(time, a_real, 'b-', label='SUMO')
ax1.plot(time, a_idm, 'g--', label='IDM')
#ax1.plot(time, a_gipps, 'r-.', label='Gipps')
ax1.plot(time, a_transworld, 'r.-', label='Transworld')
ax1.set_xlabel('Time (s)',fontsize=10)
ax1.set_ylabel('Acceleration (m/s^2)',fontsize=10)
ax1.set_title('Acceleration for front vehicle',fontsize=10)
ax1.legend(fontsize=10)

ax2.plot(time, v_real, 'b-', label='SUMO')
ax2.plot(time, v_idm, 'g--', label='IDM')
#ax2.plot(time, v_gipps, 'r-.', label='Gipps')
ax2.plot(time, v_transworld, 'r.-', label='Transworld')
ax2.set_xlabel('Time (s)',fontsize=10)
ax2.set_ylabel('Speed (m/s)',fontsize=10)
ax2.set_title('Speed for front vehicle',fontsize=10)
ax2.legend(fontsize=10)

ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)

ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)

# ax1.text(0.5, -0.1, '(a)', transform=ax1.transAxes, size=14, )
# ax2.text(0.5, -0.1, '(b)', transform=ax2.transAxes, size=14, )

plt.tight_layout()



#plt.savefig('cf_plot.svg', format='svg')

save_dir = Path(path_cwd).resolve()/ "eval"/ "figs"
plt.savefig(save_dir / 'cf_plot.svg', format='svg')
plt.savefig(save_dir / 'cf_plot.png', dpi=300, bbox_inches='tight')

plt.show()
