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

exp_dir = Path(path_cwd).resolve().parent / "experiment"
exp_dir

# training_step = 50
# pred_step = 10
# prev_step = 20
# case =  "traci_tls"
# test = "test2"
# out = "out_dim_50_n_heads_4_n_layer_4_pred_step_10"

# sim_feat = load_data(exp_dir,case,test,out,training_step,pred_step)

# real_data_dir = exp_dir / case / "data" / test / "test_data"
# train_data_dir = exp_dir / case  / "data" / test / "train_data"
# node_all = pd.read_csv(train_data_dir / "node_all.csv")
# node_id_dict = generate_unique_node_id(node_all)

# #real_struc, real_feat, node_id_dict, scalers =  load_graph(real_data_dir,0,training_step+pred_step*10,node_id_dict)
# real_struc, real_feat, node_id_dict, scalers =  load_graph(real_data_dir,0,100,node_id_dict)

# sim_feat = unscale_feat(sim_feat, scalers)
# real_feat = unscale_feat(real_feat, scalers)

def get_speed_density_volume(node_id,real_feat):
    volume = real_feat['lane'][node_id]['vehicles'].numpy().flatten()
    occupancy = real_feat['lane'][node_id]['occupancy'].numpy().flatten()
    lane_len = float(real_feat['lane'][node_id]['length'][0])
    x_step = 5
    occupancy = [sum(occupancy[i:i+x_step]) / x_step for i in range(0, len(occupancy), x_step)]
    volume = [sum(volume[i:i+x_step]) for i in range(0, len(volume), x_step)]
    density = [x * lane_len/5 for x in occupancy]
    
    speed = []
    for ai, bi in zip(volume, density):
        if bi == 0:
            speed.append(float('nan'))
        else:
            speed.append(ai / bi)

    pairs = [(density[i], speed[i], volume[i],occupancy[i]) for i in range(len(volume))]
    
    return pairs

def get_speed_density(real_feat, sim_feat):
    # Get density, speed, and volume data for real and simulated features
    pairs_real = []
    pairs_sim = []

    for node_id in real_feat['lane'].keys():
        if node_id in sim_feat['lane']:
            real_density_speed_vol_occ = get_speed_density_volume(node_id, real_feat)
            sim_density_speed_vol_occ = get_speed_density_volume(node_id, sim_feat)
            pairs_real.extend(real_density_speed_vol_occ)
            pairs_sim.extend(sim_density_speed_vol_occ)

    # Separate data by type for real features
    density_r, speeds_r, volumes_r, occupancy_r = zip(*pairs_real)

    avg_speed_r = []
    avg_density_r = []
    avg_volume_r = []

    for density_level in set(density_r):
        speeds_r_at_level = [speeds_r[i] for i in range(len(speeds_r)) if density_r[i] == density_level and not math.isnan(speeds_r[i])]
        volumes_r_at_level = [volumes_r[i] for i in range(len(volumes_r)) if density_r[i] == density_level and not math.isnan(volumes_r[i])]
        
        if len(speeds_r_at_level) > 0 and len(volumes_r_at_level) > 0:
            avg_speed_r.append(sum(speeds_r_at_level) / len(speeds_r_at_level))
            avg_density_r.append(density_level)
            avg_volume_r.append(sum(volumes_r_at_level) / len(volumes_r_at_level))

    # Separate data by type for simulated features
    density_s, speeds_s, volumes_s, occupancy_s = zip(*pairs_sim)

    avg_speed_s = []
    avg_density_s = []
    avg_volume_s = []

    for density_level in set(density_s):
        speeds_s_at_level = [speeds_s[i] for i in range(len(speeds_s)) if density_s[i] == density_level and not math.isnan(speeds_s[i])]
        volumes_s_at_level = [volumes_s[i] for i in range(len(volumes_s)) if density_s[i] == density_level and not math.isnan(volumes_s[i])]

        if len(speeds_s_at_level) > 0 and len(volumes_s_at_level) > 0:
            avg_speed_s.append(sum(speeds_s_at_level) / len(speeds_s_at_level))
            avg_density_s.append(density_level)
            avg_volume_s.append(sum(volumes_s_at_level) / len(volumes_s_at_level))

    return (avg_density_r, avg_speed_r, avg_volume_r), (avg_density_s, avg_speed_s, avg_volume_s)

def plot_MFD(real_feat, sim_feat):
    # Get speed-density data for real and simulated features
    (density_r, speeds_r, volumes_r), (density_s, speeds_s, volumes_s) = get_speed_density(real_feat, sim_feat)

    # Plot macro fundamental diagram
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Density-Speed subplot
    ax1.scatter(density_r, speeds_r, alpha=0.6, s=5,color='grey', label='Real')
    ax1.scatter(density_s, speeds_s, alpha=0.6, s=5,color='blue', label='TransWorldNG')
    ax1.set_xlabel('Density (vehicles/km)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Density-Speed MFD')
    ax1.set_ylim(0,50)
    ax1.legend()

    # Density-Volume subplot
    ax2.scatter(density_r, volumes_r, alpha=0.6, s=5, color='grey', label='Real')
    ax2.scatter(density_s, volumes_s, alpha=0.6, s=5, color='blue', label='TransWorldNG')
    ax2.set_xlabel('Density (vehicles/km)')
    ax2.set_ylabel('Volume (vehicles/h)')
    ax2.set_title('Density-Volume MFD')
    ax2.legend()

    # Volume-Speed subplot
#     ax3.scatter(volumes_r, speeds_r, alpha=0.7, color='blue', label='SUMO')
#     ax3.scatter(volumes_s, speeds_s, alpha=0.7, color='orange', label='TransWorld')
#     ax3.set_xlabel('Volume (vehicles/h)')
#     ax3.set_ylabel('Speed (km/h)')
#     ax3.set_title('Volume-Speed MFD')
#     ax3.legend()
    
    plt.tight_layout()

    plt.savefig('plot_city.png', dpi=400, format='png')

    plt.show()


training_step = 50
pred_step = 10
prev_step = 20
# case =  "bologna_clean"
# test = "test100"
# out = "out_dim_50_n_heads_4_n_layer_4_pred_step_10"

# real_data_dir = exp_dir / "bologna_clean" / "data" / "test100" / "test_data"
# train_data_dir = exp_dir / "bologna_clean" / "data" / "test100" / "train_data"
# out_dir = exp_dir / "bologna_clean" / "data" / "test100" / "out_dim_50_n_heads_4_n_layer_4_pred_step_10"


real_data_dir = exp_dir / "hangzhou" / "data" / "run1" / "test_data"
train_data_dir = exp_dir / "hangzhou" / "data" / "run1" / "train_data"
out_dir = exp_dir / "hangzhou" / "data" / "run1" / "out_dim_50_n_heads_4_n_layer_4_pred_step_10"

sim_feat = load_model_result(train_data_dir,out_dir,training_step,pred_step)

node_all = pd.read_csv(train_data_dir / "node_all.csv")
node_id_dict = generate_unique_node_id(node_all)
#real_struc, real_feat, node_id_dict, scalers =  load_graph(real_data_dir,0,training_step+pred_step*10,node_id_dict)
real_struc, real_feat, node_id_dict, scalers =  load_graph(real_data_dir,0,500,node_id_dict)

sim_feat = unscale_feat(sim_feat, scalers)
real_feat = unscale_feat(real_feat, scalers)

plot_MFD(real_feat, sim_feat)

plt.savefig('plot.png', dpi=300, format='png')
