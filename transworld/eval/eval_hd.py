import os
from pathlib import Path

#path_cwd = '/mnt/workspace/wangding/Desktop/tsim/tsim'
path_cwd = '/mnt/workspace/wangding/Desktop/TransWorldNG/transworld'
os.chdir(Path(path_cwd).absolute())
print(os.getcwd())

import matplotlib.pyplot as plt
import networkx as nx
import imageio
from pathlib import Path
from graph.load import load_graph
from eval.eval import load_model_result,unscale_feat
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
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')


# def set_working_directory(path_cwd):
#     os.chdir(Path(path_cwd).absolute())
#     print(os.getcwd())

def load_data(sumo_dir, transworld_pretrain_dir, training_step, pred_step):
    node_all = pd.read_csv(sumo_dir / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)

    sumo_struc, sumo_feat, node_id_dict, scalers = load_graph(sumo_dir, training_step, training_step + pred_step * 10, node_id_dict, scale=True)
    print('--------load sumo data----------')


    # load_data(train_data_dir,out_dir,training_step,pred_step)
    transworld_pretrain_feat = load_model_result(sumo_dir, transworld_pretrain_dir, training_step, pred_step)
    
    transworld_pretrain_feat = unscale_feat(transworld_pretrain_feat, scalers)
    sumo_feat = unscale_feat(sumo_feat, scalers)

    print('--------load transworld_pretrain_dir data--------')

    # transworld_cali_feat = load_hddata(transworld_cali_dir, training_step, pred_step)
    # print('load transworld_cali_feat data')

    # data_dir = '/mnt/workspace/wangding/Desktop/tsim/HighD/highway02/02_tracks.csv'
    # real_data = pd.read_csv(data_dir)
    # real_data['frame'] = real_data['frame'].astype(float)
    # real_data = real_data[real_data['frame'] > 0]
    # real_data = real_data[real_data['frame'] < 1100]

    # print('load real data')
    return sumo_feat, transworld_pretrain_feat

def plot_histogram(sumo_feat, transworld_pretrain_feat):
    sns.set(style='whitegrid', palette='dark')

    # Extract the desired feature from the sim_feat and real_feat dictionaries for all vehicles
    #[xVelocity,speed]
    #[xAcceleration,acceleration]
    my_feature_transworld_preTrain = [transworld_pretrain_feat['veh'][veh_id]['speed'] for veh_id in transworld_pretrain_feat['veh']]
    all_values_transworld_preTrain= torch.cat(my_feature_transworld_preTrain, dim=0)
    all_values_transworld_preTrain = all_values_transworld_preTrain.numpy()

    my_feature_sumo = [sumo_feat['veh'][veh_id]['speed'] for veh_id in sumo_feat['veh']]
    all_values_sumo = torch.cat(my_feature_sumo, dim=0)
    all_values_sumo = all_values_sumo.numpy()

    # Set up the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the histograms using seaborn for each dataset
    sns.distplot(all_values_transworld_preTrain, ax=ax, label='Transworld_preTrain', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})
    sns.distplot(all_values_sumo, ax=ax, label='sumo_feat', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

    # Add a legend and axis labels
    ax.legend()
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Density')
    ax.set_title('Comparision among Real Data, TransWorld')
    plt.savefig('histogram_all.png')
    plt.show()


exp_dir = Path(path_cwd).resolve().parent
# Load data
sumo_dir = exp_dir / "experiment" / "hangzhou" / "data" / "run1" / "train_data"
transworld_pretrain_dir = exp_dir / "experiment" / "hangzhou" / "data" / "run1" / "out_dim_50_n_heads_4_n_layer_4_pred_step_10"
training_step = 50
pred_step = 10
sumo_feat, transworld_pretrain_feat = load_data(sumo_dir, transworld_pretrain_dir, training_step, pred_step)

# Plot histogram
plot_histogram(sumo_feat, transworld_pretrain_feat)






# sumo_dir = exp_dir/ "experiment" / "hangzhou" / "data" / "run1" / "train_data"
# print(sumo_dir)

# node_all = pd.read_csv(sumo_dir / "node_all.csv")
# node_id_dict = generate_unique_node_id(node_all)

# training_step = 50
# pred_step = 10

# sumo_struc, sumo_feat, node_id_dict, scalers =  load_graph(sumo_dir,training_step,training_step+pred_step*10,node_id_dict)
# print('load sumo data')

# transworld_pretrain_dir = exp_dir / "experiment" / "hangzhou" / "data" / "run1"  /  "out_dim_50_n_heads_4_n_layer_4_pred_step_10"
# #transworld_cali_dir = exp_dir / "HighD" / 'highway02' / 'FineTuneModel' / "out_dim_100_n_heads_4_n_layer_4_pred_step_10"


# transworld_pretrain_feat = load_hddata(transworld_pretrain_dir,training_step,pred_step)
# print('load transworld_pretrain_dir data')

# # transworld_cali_feat = load_hddata(transworld_cali_dir,training_step,pred_step)
# # print('load transworld_cali_feat data')

# # data_dir = '/mnt/workspace/wangding/Desktop/tsim/HighD/highway02/02_tracks.csv'
# # real_data = pd.read_csv(data_dir)
# # real_data['frame'] = real_data['frame'].astype(float)
# #real_data = real_data[real_data['frame']>0]
# #real_data = real_data[real_data['frame']<1100]

# # print('load real data')

# #/mnt/workspace/wangding/Desktop/tsim/HighD/highway02/02_tracks.csv
# ##########################################
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set(style='whitegrid', palette='dark')
# # print(sumo_feat.keys(),transworld_feat.keys())
# # Extract the desired feature from the sim_feat and real_feat dictionaries for all vehicles
# #[xVelocity,speed]
# #[xAcceleration,acceleration]
# my_feature_transworld_preTrain = [transworld_pretrain_feat['veh'][veh_id]['acceleration'] for veh_id in transworld_pretrain_feat['veh']]
# all_values_transworld_preTrain= torch.cat(my_feature_transworld_preTrain, dim=0)
# all_values_transworld_preTrain = all_values_transworld_preTrain.numpy()

# # my_feature_transworld_cali = [transworld_cali_feat['veh'][veh_id]['xAcceleration'] for veh_id in transworld_cali_feat['veh']]
# # all_values_transworld_cali= torch.cat(my_feature_transworld_cali, dim=0)
# # all_values_transworld_cali = all_values_transworld_cali.numpy()

# my_feature_sumo = [sumo_feat['veh'][veh_id]['speed'] for veh_id in sumo_feat['veh']]
# all_values_sumo = torch.cat(my_feature_sumo, dim=0)
# all_values_sumo = all_values_sumo.numpy()

# # Set up the figure and axis objects
# fig, ax = plt.subplots(figsize=(10, 5))

# # Plot the histograms using seaborn for each dataset

# sns.distplot(all_values_transworld_preTrain, ax=ax, label='Transworld_preTrain', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})
# sns.distplot(all_values_sumo, ax=ax, label='sumo_feat', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})
# #sns.distplot(real_data['acceleration'], ax=ax, label='Real',hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

# # Add a legend and axis labels
# ax.legend()
# ax.set_xlabel('Velocity')
# ax.set_ylabel('Density')
# ax.set_title('Comparision among Real Data, TransWorld')
# plt.savefig('histogram_all.png')

# # Show the plot
# plt.show()