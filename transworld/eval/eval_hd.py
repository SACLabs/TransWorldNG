import os
from pathlib import Path

path_cwd = '/mnt/workspace/wangding/Desktop/TransWorldNG/transworld'
os.chdir(Path(path_cwd).absolute())
print(os.getcwd())

import matplotlib.pyplot as plt
import networkx as nx
import imageio
from pathlib import Path
from graph.load_hd import load_graph
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

def load_data(sumo_dir, transworld_pretrain_dir, transworld_cali_dir, real_data_dir, node_name, feat_name, real_feat_name, training_step, pred_step):

    if sumo_dir is not None:
        node_all = pd.read_csv(sumo_dir / "node_all.csv")
        node_id_dict = generate_unique_node_id(node_all)

        # sumo_struc_dict, sumo_feat_dict, node_id_dict, scalers = load_graph(sumo_dir, training_step, training_step + pred_step * 10, node_id_dict, scale=True)
        # sumo_feat_dict = unscale_feat(sumo_feat_dict, scalers)
        # sumo_feat = [sumo_feat_dict[node_name][idx][feat_name] for idx in sumo_feat_dict[node_name]]
        # sumo_feat = torch.cat(sumo_feat, dim=0)
        # sumo_feat = sumo_feat.numpy()


        sumo_struc_dict, sumo_feat_dict, node_id_dict, scalers = load_graph(sumo_dir, 0, 1500, node_id_dict)
        #sumo_feat_dict = unscale_feat(sumo_feat_dict, scalers)
        sumo_feat = [sumo_feat_dict[node_name][idx][feat_name] for idx in sumo_feat_dict[node_name]]
        sumo_feat = torch.cat(sumo_feat, dim=0)
        sumo_feat = sumo_feat.numpy()
        print('--------load sumo data----------')
        
    if sumo_dir is None:
        sumo_feat = None

    if transworld_pretrain_dir is not None:
        # load_data(train_data_dir,out_dir,training_step,pred_step)
        transworld_pretrain_feat_dict = load_model_result(sumo_dir, transworld_pretrain_dir, training_step, pred_step)    
        transworld_pretrain_feat_dict = unscale_feat(transworld_pretrain_feat_dict, scalers)
        transworld_pretrain_feat = [transworld_pretrain_feat_dict[node_name][idx][feat_name] for idx in transworld_pretrain_feat_dict[node_name]]
        transworld_pretrain_feat = torch.cat(transworld_pretrain_feat, dim=0)
        transworld_pretrain_feat = transworld_pretrain_feat.numpy()
        print('--------load transworld_pretrain_model results--------')

    if transworld_pretrain_dir is None:
        transworld_pretrain_feat = None
    

    if transworld_finetune_dir is not None:
        transworld_finetune_feat_dict = load_model_result(sumo_dir,transworld_finetune_dir, training_step, pred_step)
        transworld_finetune_feat_dict = unscale_feat(transworld_finetune_feat_dict, scalers)
        transworld_finetune_feat = [transworld_finetune_feat_dict[node_name][idx][feat_name] for idx in transworld_finetune_feat_dict[node_name]]
        transworld_finetune_feat= torch.cat(transworld_finetune_feat, dim=0)
        transworld_finetune_feat = transworld_finetune_feat.numpy()
        print('--------load transworld_finetue_model results--------')

    if transworld_finetune_dir is None:
        transworld_finetune_feat = None

    if real_data_dir is not None:
    # data_dir = '/mnt/workspace/wangding/Desktop/tsim/HighD/highway02/02_tracks.csv'
        real_data = pd.read_csv(real_data_dir)
        real_data['frame'] = real_data['frame'].astype(float)
        real_data = real_data[real_data['frame'] > 0]
        real_data = real_data[real_data['frame'] < 5000]
        real_feat = real_data[real_feat_name]

    # print('load real data')
    return sumo_feat, transworld_pretrain_feat, transworld_finetune_feat, real_feat

def plot_histogram(sumo_feat, transworld_pretrain_feat, transworld_finetune_feat, real_feat):
    sns.set(style='whitegrid', palette='dark')

    # Extract the desired feature from the sim_feat and real_feat dictionaries for all vehicles
    #[xVelocity,speed]
    #[xAcceleration,acceleration]
    # Set up the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the histograms using seaborn for each dataset
    if sumo_feat is not None:
        sns.distplot(sumo_feat, ax=ax, label='sumo_feat', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

    if transworld_pretrain_feat is not None:
        sns.distplot(transworld_pretrain_feat, ax=ax, label='transworld_pretrain_feat', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

    if transworld_finetune_feat is not None:
        sns.distplot(transworld_finetune_feat, ax=ax, label='Transworld_finetune', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

    if real_feat is not None:
        sns.distplot(real_feat, ax=ax, label='real_feat', hist_kws={'alpha': 0.8}, kde_kws={'linewidth': 3, 'shade': False})

    # Add a legend and axis labels
    ax.legend()
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Density')
    ax.set_title('Comparision among Real Data, TransWorld')
    plt.savefig(exp_dir/ 'transworld' / 'eval' / 'figs' / 'histogram_all.png')
    plt.show()


exp_dir = Path(path_cwd).resolve().parent
# Load data

# real_data_dir=None
# sumo_dir = exp_dir / "experiment" / "hangzhou" / "data" / "test500" / "train_data"
# transworld_pretrain_dir = exp_dir / "experiment" / "hangzhou" / "data" / "test500" / "pretain" /  "out_dim_50_n_heads_4_n_layer_4_pred_step_10"
# transworld_finetune_dir = exp_dir / "experiment" / "hangzhou" / "data" / "test500" / "finetune" /  "out_dim_50_n_heads_4_n_layer_4_pred_step_10"

real_data_dir=exp_dir / "experiment" / "HighD" / "data" / "highway02" / "02_tracks.csv"
sumo_dir = exp_dir / "experiment" / "HighD" / "data" / "highway02" / "data"
transworld_pretrain_dir = exp_dir / "experiment" / "HighD" / "data" / "highway02" / "preTrainModel" /  "out_dim_100_n_heads_4_n_layer_4_pred_step_10"
transworld_finetune_dir = exp_dir / "experiment" / "HighD" / "data" / "highway02" / "FineTuneModel" /  "out_dim_100_n_heads_4_n_layer_4_pred_step_10"

node_name ="veh"
feat_name = "yVelocity"
real_feat_name = "yVelocity"

training_step = 50
pred_step = 10

sumo_feat, transworld_pretrain_feat, transworld_finetune_feat, real_feat = load_data(sumo_dir, transworld_pretrain_dir, transworld_finetune_dir, real_data_dir, node_name, feat_name, real_feat_name, training_step, pred_step)

# Plot histogram
plot_histogram(sumo_feat, transworld_pretrain_feat, transworld_finetune_feat, real_feat)




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