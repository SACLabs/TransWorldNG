from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
from typing import Dict, Tuple
import networkx as nx
import pandas as pd
import os
import numpy as np
from graph.process import generate_unique_node_id
import pickle
from graph.load import load_graph
#import cv2
import pickle
import torch
import math
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
from collections import defaultdict
#import cv2

def load_sim_feat(out_dir,file_name):    
    with open(out_dir / file_name, "rb") as g:
        model_graph = pickle.load(g)
    unique_key = list(model_graph.keys())[0]
    sim_struc, sim_feat = model_graph[unique_key]
    return sim_feat

def load_data(exp_dir,case,test,out,training_step,pred_step):
    exp_setting = exp_dir / case
    train_data_dir = exp_setting   / "data" / test / "train_data"
    real_data_dir = exp_setting / "data" / test / "test_data"
    out_dir = exp_setting / "data" / test / out
    node_all = pd.read_csv(train_data_dir / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)
    
    file_list = os.listdir(out_dir)
    file_list = [file_name for file_name in file_list if file_name.endswith(".p")]
    
    sim_feat0 = load_sim_feat(out_dir,file_list[0])
    sim_feat1 = load_sim_feat(out_dir,file_list[1])
    sim_feat2 = load_sim_feat(out_dir,file_list[2])
    sim_feat3 = load_sim_feat(out_dir,file_list[3])
    sim_feat4 = load_sim_feat(out_dir,file_list[4])
    sim_feat5 = load_sim_feat(out_dir,file_list[5])
    sim_feat6 = load_sim_feat(out_dir,file_list[6])
    sim_feat7 = load_sim_feat(out_dir,file_list[7])
    sim_feat8 = load_sim_feat(out_dir,file_list[8])
    sim_feat9 = load_sim_feat(out_dir,file_list[9])
    
    sub_feat = [sim_feat0, sim_feat1, sim_feat2, sim_feat3, sim_feat4, sim_feat5, sim_feat6, sim_feat7, sim_feat8, sim_feat9]

    for i, sim_feat in enumerate(sub_feat):
        time_start = training_step + pred_step * (i)
        time_end = time_start + pred_step
        for node_type in sim_feat:
            for node_id in sim_feat[node_type]:
                for feat_name in sim_feat[node_type][node_id]:
                    if feat_name == "time":
                        sim_feat[node_type][node_id][feat_name] = torch.arange(time_start, time_start + len(sim_feat[node_type][node_id][feat_name]))
                        
    # sub_feat =[sim_feat0,sim_feat1,sim_feat2,sim_feat3,sim_feat4]
    n = len(sub_feat)
    #sim_feat_list = [sim_feat1,sim_feat2,sim_feat3,sim_feat4] 
    combined_feat = defaultdict(dict)
    for node_type in set([key for sim_feat in sub_feat for key in sim_feat.keys()]):
        for node_id in set([key for sim_feat in sub_feat for key in sim_feat.get(node_type, {}).keys()]):
            combined_feat[node_type][node_id] = {}
            for feat_name in set([key for sim_feat in sub_feat for key in sim_feat.get(node_type, {}).get(node_id, {}).keys()]):
                feat_list = []
                for sim_feat in sub_feat:
                    feat_list.append(sim_feat.get(node_type, {}).get(node_id, {}).get(feat_name, torch.empty(0)))
                feat_value = torch.cat(feat_list, dim=0).squeeze()
                combined_feat[node_type][node_id][feat_name] = feat_value
    sim_feat = combined_feat
    return sim_feat

def unscale_feat(feat_dict, scalers):
    """
    Unscale the scaled features in sim_feat using the corresponding scalers.

    Parameters:
    sim_feat (dict): a dictionary that contains the simulated features for each node
    scalers (dict): a dictionary that contains the scaler objects for each node type
    scale_veh_col (list): a list of columns that were scaled for vehicles
    scale_lane_col (list): a list of columns that were scaled for lanes

    Returns:
    sim_feat (dict): the updated dictionary with unscaled features
    """
    scale_veh_col = ['acceleration','coor_x','coor_y','length','pos_on_lane','speed','tlc_state','x']
    scale_lane_col = ['length','occupancy','shape_a','shape_b','shape_c','shape_d','vehicles']

    for node_type, node_data in feat_dict.items():
        # get the respective scaler object for the node type
        scaler = scalers[node_type]

        # loop through each node id for the node type
        for node_id, node_feat in node_data.items():
            # get the columns that were scaled for the node id
            if node_type == 'veh':
                scale_col = scale_veh_col
            elif node_type == 'lane':
                scale_col = scale_lane_col
            else:
                continue
            
            # replace the scaled values with the unscaled values
            for col in node_feat.keys():
                if col in scale_col:
                    #print(col)
                    idx = scale_col.index(col)
                    min_val = scalers[node_type].data_min_[idx]
                    max_val = scalers[node_type].data_max_[idx]
                    feat_dict[node_type][node_id][col] =  feat_dict[node_type][node_id][col]*(max_val -min_val) + min_val
    return feat_dict


def get_feat_df(feat_dict, node_type, node_id, feat):
    feat_lst = feat_dict[node_type][node_id][feat].tolist()
    time_lst = feat_dict[node_type][node_id]['time'].tolist() 
    #print(feat_lst)
    g_feat = pd.DataFrame(np.array([feat_lst, time_lst]).T)
    g_feat.columns = [feat+str(node_id), 'time']
    g_feat['time'] = [int(i) for i in g_feat.time]
    return g_feat


# def plot_feat(real_feat,sim_feat,node_type,node_id,feat_name):
#     feat_real = get_feat_df(real_feat, node_type, node_id, feat_name)
#     feat_sim = get_feat_df(sim_feat, node_type, node_id, feat_name)
#     feat_sim = feat_sim[feat_sim['time'] != 0]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
#     ax.plot(feat_real['time'],feat_real[feat_name+str(node_id)], label = "real")
#     ax.plot(feat_sim['time'],feat_sim[feat_name+str(node_id)], linestyle='--',  marker='o',
#         markersize=2, label = "sim")
#     ax.set_xlabel("steps")
#     ax.set_ylabel(feat_name)
#     #ax.set_ylim(0,1)
#     ax.set_title( feat_name+' of '+ node_type+'_'+str(node_id))
#     ax.legend()



def get_cf_df(struc_dict):
    front_veh = struc_dict[('veh', 'phy/ahead', 'veh')][0].tolist()
    behind_veh = struc_dict[('veh', 'phy/ahead', 'veh')][1].tolist()
    time = struc_dict[('veh', 'phy/ahead', 'veh')][2].tolist()
    cf_struc = pd.DataFrame(np.array([front_veh, behind_veh,time]).T)
    cf_struc.columns = ['front_veh', 'behind_veh', 'time']
    return cf_struc
# def CF_plot(real_struc, real_feat_dict,sim_feat_dict,front_veh_id,behind_veh_id,feat_name,plot_cf):
#     cf_struc = get_cf_df(real_struc)
#     veh_pair = cf_struc[(cf_struc['front_veh'] == front_veh_id) & (cf_struc['behind_veh'] == behind_veh_id)]

#     feat_front = get_feat_df(real_feat_dict, 'veh', front_veh_id, feat_name)
#     feat_behind = get_feat_df(real_feat_dict, 'veh', behind_veh_id, feat_name)
#     feat_front_sim = get_feat_df(sim_feat_dict, 'veh', front_veh_id, feat_name)
#     feat_behind_sim = get_feat_df(sim_feat_dict, 'veh', behind_veh_id,feat_name)

#     merge_front = pd.merge(veh_pair, feat_front, on='time')
#     merge_behind = pd.merge(merge_front, feat_behind, on='time')
#     merge_front_sim = pd.merge(veh_pair, feat_front_sim, on='time')
#     merge_behind_sim = pd.merge(merge_front_sim, feat_behind_sim, on='time')
    
#     if plot_cf == 1:
#         plt.plot(merge_behind['time'], merge_behind[feat_name+str(front_veh_id)], label = "Front_car")
#         plt.plot(merge_behind['time'], merge_behind[feat_name+str(behind_veh_id)], label = "Follower_car")
#         plt.plot(merge_behind_sim['time'], merge_behind_sim[feat_name+str(front_veh_id)], linestyle='--',  marker='o', markersize=2, label = "Front_car_sim")
#         plt.plot(merge_behind_sim['time'],merge_behind_sim[feat_name+str(behind_veh_id)], linestyle='--',  marker='o',
#             markerfacecolor='blue', markersize=2, label = "Follower_car_sim")
#         plt.xlabel("Time step")
#         plt.ylabel(feat_name.capitalize())
#         plt.title(feat_name.capitalize() + " of vehicle pairs")
#         plt.legend(fontsize=10)
#         plt.show()
#     else:
#         return merge_behind_sim
    
    
    
    
    
    
    
    
    

# def get_real_feat(real_feat, node_type, node_id, feat):
#     feat_lst = real_feat[node_type][node_id][feat].tolist()
#     time_lst = real_feat[node_type][node_id]['time'].tolist() 
#     g_feat = pd.DataFrame(np.array([feat_lst, time_lst]).T)
#     g_feat.columns = [feat+str(node_id), 'time']
#     g_feat['time'] = [int(i) for i in g_feat.time]
#     return g_feat


# def get_sim_feat(model_graph, node_type, node_id, feat):
#     node_name = node_type+'/'+ str(node_id)
#     for keys in model_graph:
#         name = keys.split('@')[0]
#         if node_name == name:
#             feat_dict = model_graph[keys][1]
#             feat_lst = feat_dict[node_type][node_id][feat].tolist()
#             time_lst = feat_dict[node_type][node_id]['time'].tolist() 
#             g_feat = pd.DataFrame(np.array([feat_lst[0], time_lst[0]]).T)
#             g_feat.columns = [feat+str(node_id), 'time']
#             g_feat['time'] = [int(i) for i in g_feat.time]
#     return g_feat


# def model_struc_to_eval(model_graph: Dict)->pd.DataFrame:
#     struc_df = pd.DataFrame()
#     row = []
#     for seed_node in model_graph.keys():
#         struc_dict = model_graph[seed_node][0]
#         for key, values in struc_dict.items():
#             sr_type, rela, end_type = key
#             sr_node, end_node, time = values
#             row.append([sr_type, rela, end_type,int(sr_node[-1]),int(end_node[-1]),int(time[-1])])
#             #for i in range(len(time)):
#                 #row.append([sr_type, rela, end_type,int(sr_node[i]),int(end_node[i]),int(time[i])])
    
#         edge = pd.DataFrame(row, columns=['sr_type', 'rela', 'end_type','sr_node','end_node','step'])
#         edge["from"] = [edge["sr_type"][i] + str(edge["sr_node"][i]) for i in range(len(edge["sr_node"]))]
#         edge["to"] = [edge["end_type"][i] + str(edge["end_node"][i]) for i in range(len(edge["end_node"]))]
#         edge = edge[["step","from","to", 'rela']]
#         struc_df = pd.concat([struc_df,edge])
#     return struc_df


# def model_struc_to_eval1(model_graph: Dict)->pd.DataFrame:
#     struc_df = pd.DataFrame()
#     row = []
#     for seed_node in model_graph.keys():
#         struc_dict = model_graph[seed_node][0]
#         for key, values in struc_dict.items():
#             sr_type, rela, end_type = key
#             sr_node, end_node, time = values
#             for i in range(len(time)):
#                 if int(time[i]) == 100:
#                     row.append([sr_type, rela, end_type,int(sr_node[i]),int(end_node[i]),int(time[i])])
#         edge = pd.DataFrame(row, columns=['sr_type', 'rela', 'end_type','sr_node','end_node','step'])
#         edge["from"] = [edge["sr_type"][i] + str(edge["sr_node"][i]) for i in range(len(edge["sr_node"]))]
#         edge["to"] = [edge["end_type"][i] + str(edge["end_node"][i]) for i in range(len(edge["end_node"]))]
#         edge = edge[["step","from","to", 'rela']]
#         struc_df = pd.concat([struc_df,edge])
#     return struc_df


# def raw_struc_to_eval(struc_dict: Dict)-> pd.DataFrame:
#     row = []
#     for key, values in struc_dict.items():
#         sr_type, rela, end_type = key
#         sr_node, end_node, time = values
#         for i in range(len(time)):
#             row.append([sr_type, rela, end_type,int(sr_node[i]),int(end_node[i]),int(time[i])])
#     raw_df = pd.DataFrame(row, columns=['sr_type', 'rela', 'end_type','sr_node','end_node','step'])
#     raw_df["from"] = [raw_df["sr_type"][i] + str(raw_df["sr_node"][i]) for i in range(len(raw_df["sr_node"]))]
#     raw_df["to"] = [raw_df["end_type"][i] + str(raw_df["end_node"][i]) for i in range(len(raw_df["end_node"]))]
#     raw_df = raw_df[["step","from","to", 'rela']]
#     return raw_df   


# def model_feat_to_eval(model_graph, sim_struc, node_type, node_feat):
#     feat = []
#     for key in model_graph.keys():
#         node_name = key.split('@')[0]
#         type, node_id = node_name.split('/')
#         if node_type in node_name:
#             #print(node_name, model_graph[key][1]['veh'][node_id]['speed'])
#             value = model_graph[key][1][node_type][int(node_id)][node_feat][0].tolist()
#             feat.append([node_type+node_id, value])

#     filter_from = sim_struc[sim_struc['from'].str.contains(node_type)]
#     filter_to = filter_from[filter_from['to'].str.contains(node_type)]

#     df_feat = pd.DataFrame(feat,columns=['node_name', node_feat])
#     merge_from = pd.merge(df_feat, filter_to, left_on='node_name', right_on='from')
#     merge_from = merge_from[['step','from','to','rela',node_feat]]
#     merge_to = pd.merge(df_feat, merge_from, left_on='node_name', right_on='to')
#     merge_to.rename(columns={node_feat+'_x':'from_'+ node_feat,node_feat+'_y':'to_'+ node_feat}, inplace=True)
#     merge_to = merge_to[['step','from','to','rela','from_'+ node_feat,'to_'+ node_feat]]
#     return merge_to   

# def get_struc_df(real_struc):
#     front_veh = real_struc[('veh', 'phy/ahead', 'veh')][0].tolist()
#     behind_veh = real_struc[('veh', 'phy/ahead', 'veh')][1].tolist()
#     time = real_struc[('veh', 'phy/ahead', 'veh')][2].tolist()
#     g_struc = pd.DataFrame(np.array([front_veh, behind_veh,time]).T)
#     g_struc.columns = ['front_veh', 'behind_veh', 'time']
#     return g_struc


# def get_cf_veh_pair(g_struc, front_veh_id, behind_veh_id):
#     veh_pair = g_struc[(g_struc['front_veh'] == front_veh_id) & (g_struc['behind_veh'] == behind_veh_id)]
#     return veh_pair


# def compare_struc(root: str, real_struc: Dict, sim_struc: Dict, out_dir: Path, *kwargs) -> None:
#     for t in range(1,20):
#         sub_real = real_struc[(real_struc["step"] == t) & (real_struc["from"]== root)]
#         real_G = nx.from_pandas_edgelist(sub_real, 'from', 'to')
       
#         sub_sim = sim_struc[(sim_struc["step"] == t) & (sim_struc["from"]== root)]
#         sim_G = nx.from_pandas_edgelist(sub_sim, 'from', 'to')

#         fig, axs = plt.subplots(ncols=2, figsize=(14, 4))
#         plt.figtext(0.5, 1, "step={s}".format(s=t), ha='center', va='top')
#         nx.draw_circular(real_G, ax=axs[0], with_labels=True)
#         nx.draw_circular(sim_G, ax=axs[1], with_labels=True)

#         plt.savefig(out_dir / "step_{s}.png".format(s=t), format="PNG",)
#         plt.close()
    
# def get_struc_df(real_struc):
#     front_veh = real_struc[('veh', 'phy/ahead', 'veh')][0].tolist()
#     behind_veh = real_struc[('veh', 'phy/ahead', 'veh')][1].tolist()
#     time = real_struc[('veh', 'phy/ahead', 'veh')][2].tolist()
#     g_struc = pd.DataFrame(np.array([front_veh, behind_veh,time]).T)
#     g_struc.columns = ['front_veh', 'behind_veh', 'time']
#     return g_struc
# def get_cf_veh_pair(g_struc, front_veh_id, behind_veh_id):
#     veh_pair = g_struc[(g_struc['front_veh'] == front_veh_id) & (g_struc['behind_veh'] == behind_veh_id)]
#     return veh_pair

# def get_real_feat(real_feat, node_type, node_id, feat):
#     feat_lst = real_feat[node_type][node_id][feat].tolist()
#     time_lst = real_feat[node_type][node_id]['time'].tolist() 
#     #print(feat_lst)
#     g_feat = pd.DataFrame(np.array([feat_lst, time_lst]).T)
#     g_feat.columns = [feat+str(node_id), 'time']
#     g_feat['time'] = [int(i) for i in g_feat.time]
#     return g_feat

# def get_sim_feat(model_graph, node_type, node_id, feat):
#     node_name = node_type+'/'+ str(node_id)
#     g_feat = pd.DataFrame()
#     for keys in model_graph:
#         name = keys.split('@')[0]
#         if node_name == name:
#             feat_dict = model_graph[keys][1]
#             feat_lst = feat_dict[node_type][node_id][feat].tolist()
#             time_lst = feat_dict[node_type][node_id]['time'].tolist() 
            
#             feat_lst = [item for sublist in feat_lst for item in sublist]
#             #time_lst = [item for sublist in time_lst for item in sublist]
#             #print(len(feat_lst), len(time_lst))
#             #print(feat_lst, time_lst)

#             g_feat = pd.DataFrame(np.array([feat_lst, time_lst]).T)
#             g_feat.columns = [feat+str(node_id), 'time']
#             g_feat['time'] = [int(i) for i in g_feat.time]
#     if not g_feat.empty:
#         return g_feat
#     else:
#         print("g_feat is empty")


# def feat_plot(node_type, node_id, feat_name):
#     feat_real = get_real_feat(real_feat, node_type, node_id, feat_name)
#     feat_sim = get_sim_feat(model_graph, node_type, node_id, feat_name)
#     plt.plot(feat_real['time'], feat_real[feat_name+str(node_id)], label = "real")
#     plt.plot(feat_sim['time'], feat_sim[feat_name+str(node_id)], linestyle='--',  marker='o',
#         markersize=2, label = "sim")
    
#     plt.xlabel("Time step")
#     plt.ylabel(feat_name)
#     plt.title(feat_name + "_of_" + node_type + str(node_id))
#     plt.legend()

# def plot_avg_feat(real_feat, model_graph, node_type, feat_name):
#     feat_lst_real = []
#     feat_lst_sim = []
#     avg_feat_real = pd.DataFrame()
#     avg_feat_sim = pd.DataFrame()
#     for node_id in list(real_feat[node_type].keys()):
#         df_real = get_real_feat(real_feat, node_type, node_id, feat_name) 
#         df_sim = get_sim_feat(model_graph, node_type, node_id, feat_name)
        
#         feat_lst_real.append(df_real[feat_name+str(node_id)])
#         feat_lst_sim.append(df_sim[feat_name+str(node_id)])
    
#     df_feat_real = pd.DataFrame(np.array(feat_lst_real))
#     df_feat_sim = pd.DataFrame(np.array(feat_lst_sim))
    
#     feat_real_avg = df_feat_real.mean()
#     feat_sim_avg = df_feat_sim.mean()
    
#     time_real =  df_real['time']
#     time_sim =  df_sim['time']
    
#     avg_feat_real['time'] = time_real
#     avg_feat_sim['time'] = time_sim
#     avg_feat_real[feat_name+'_avg'] = feat_real_avg
#     avg_feat_sim[feat_name+'_avg'] = feat_sim_avg
    
#     plt.plot(avg_feat_real['time'], avg_feat_real[feat_name+'_avg'], label = "real")
#     plt.plot(avg_feat_sim['time'], avg_feat_sim[feat_name+'_avg'], label = "sim")

#     plt.xlabel("Time step")
#     plt.ylabel(feat_name)
#     plt.title("Citywide_avg_"+ node_type + "_" + feat_name)
#     plt.legend()
#     plt.show()
#     # return avg_feat_real, avg_feat_sim
    
# def CF_plot(front_veh_id,behind_veh_id,feat_name):
#     g_struc = get_struc_df(real_struc)
#     veh_pair = get_cf_veh_pair(g_struc, front_veh_id, behind_veh_id)

#     feat_front = get_real_feat(real_feat, 'veh', front_veh_id, feat_name)
#     feat_behind = get_real_feat(real_feat, 'veh', behind_veh_id, feat_name)
#     feat_front_sim = get_sim_feat(model_graph, 'veh', front_veh_id, feat_name)
#     feat_behind_sim = get_sim_feat(model_graph, 'veh', behind_veh_id,feat_name)

#     merge_front = pd.merge(veh_pair, feat_front, on='time')
#     merge_behind = pd.merge(merge_front, feat_behind, on='time')
#     merge_front_sim = pd.merge(veh_pair, feat_front_sim, on='time')
#     merge_behind_sim = pd.merge(merge_front_sim, feat_behind_sim, on='time')

#     plt.plot(merge_behind['time'], merge_behind[feat_name+str(front_veh_id)], label = "Front_car")
#     plt.plot(merge_behind['time'], merge_behind[feat_name+str(behind_veh_id)], label = "Follower_car")
#     plt.plot(merge_behind_sim['time'], merge_behind_sim[feat_name+str(front_veh_id)], linestyle='--',  marker='o',
#         markerfacecolor='blue', markersize=2, label = "Front_car_sim")
#     plt.plot(merge_behind_sim['time'],merge_behind_sim[feat_name+str(behind_veh_id)], linestyle='--',  marker='o',
#         markerfacecolor='blue', markersize=2, label = "Follower_car_sim")
#     plt.xlabel("Time step")
#     plt.ylabel(feat_name)
#     plt.title(feat_name + " of vehicle pairs")
#     plt.legend()
#     plt.show()
    