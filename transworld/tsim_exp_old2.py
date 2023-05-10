from pathlib import Path
from graph.load import load_graph
from rules.pre_process import load_veh_depart, pre_actions
from rules.post_process import load_veh_route, post_actions
import random
import torch.nn as nn
from game.model import HGT, RuleBasedGenerator, GraphStateLoss
from game.data import DataLoader, Dataset
from game.graph import Graph
import game
import matplotlib.pyplot as plt
import torch
from game.operator.transform import dgl_graph_to_graph_dict
from tqdm import tqdm
from datetime import datetime
#from eval.eval import struc_dict_to_eval, raw_edge_to_eval, compare_feat, compare_struc
import pickle
import os
import csv
import argparse
import logging
import shutil
import sys
import multiprocessing
from multiprocessing import Pool
random.seed(3407)

def print_loss(epoch, i, loss):
    with open('loss.csv','a') as loss_log:
        train_writer = csv.writer(loss_log)
        train_writer.writerow([str(epoch), str(i), str(round(loss,4))])

def train(timestamps, graph, batch_size, encoder, generator, veh_route, loss_fcn, optimizer, logger):
    logger.info("========= start generate dataset =======")
    train_dataset = Dataset(timestamps, train_mode=True)
    train_loader = DataLoader(train_dataset, graph.operate, batch_size=batch_size, num_workers=0)
    logger.info("========== finish generate dataset =======")
    # graph_dicts = {}
    logger.info("========= start training =======")
    for i, (cur_graphs, next_graphs) in tqdm(enumerate(train_loader)):  # 这里for的是时间戳
        # 下面这个for循环后面会改成并发操作
        loss = 0.
        for ((_, cur_graph), (seed_node_n_time, next_graph)) in zip(cur_graphs.items(), next_graphs.items()):  
            assert (_.split("@")[0]) == seed_node_n_time.split("@")[0], ValueError("Dataloader Error! node_name not equal")
            node_type = seed_node_n_time.split("/")[0]
            if node_type == 'veh':
                continue
            time = float(seed_node_n_time.split("@")[1])
            node_repr = encoder(cur_graph)
            actions, pred_graph = generator([seed_node_n_time], cur_graph, node_repr, veh_route)
            loss = loss + loss_fcn(pred_graph, next_graph)
        logger.info(f"------------ current_loss is {loss.item()} ---------")
        # print_loss(epoch, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info("========== finished training ==========")

@torch.no_grad()
def eval(graph, batch_size, encoder, generator, veh_depart, veh_route, changable_feature_names, hetero_feat_dim):
    val_timestamp = [float(i) for i in range(300,319)]
    val_dataset = Dataset(val_timestamp, train_mode=False)
    val_loader = DataLoader(val_dataset, graph.operate, batch_size=batch_size, num_workers=0)

    for i, cur_graphs in tqdm(enumerate(val_loader)):
        val_result = {}
        for seed_node_n_time, cur_graph in cur_graphs.items():
            node_type = seed_node_n_time.split("/")[0]
            if node_type == 'veh':
                continue
            time = float(seed_node_n_time.split("@")[1])
            actions_pre = pre_actions(veh_depart, time, cur_graph)
            node_repr = encoder(cur_graph)
            actions, pred_graph = generator(
                [seed_node_n_time], cur_graph, node_repr, veh_route
            )            
            graph.states_to_feature_event(time, changable_feature_names, cur_graph, pred_graph)
            graph.actions_to_game_operations(actions_pre)
            if actions != {}:
                graph.actions_to_game_operations(actions)
                print(actions)
                
    total_graphs = val_loader.collate_tool([max(val_timestamp)+1.], max_step=None)
    for seed_node_n_time, total_graph in total_graphs[0].items():
        struc_dict, feat_dict = dgl_graph_to_graph_dict(total_graph, hetero_feat_dim)
        val_result[seed_node_n_time] = (struc_dict, feat_dict)
    return val_result      


def create_folder(folder_path, delete_origin=False):
    # 这个函数的作用是,当文件夹存在就删除,然后重新创建一个新的
    if not os.path.exists(folder_path):
        # shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
    else:
        if delete_origin:
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

def setup_logger(name, log_folder_path, level=logging.DEBUG):
    create_folder(log_folder_path)
    log_file = log_folder_path /f"{name}_log"
    handler = logging.FileHandler(log_file,encoding="utf-8",mode="a")
    formatter = logging.Formatter("%(asctime)s,%(msecs)d,%(levelname)s,%(name)s::%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def run(dimension, n_heads, n_layer):
    time_diff = []
    #for test in [1,2,3,4,5]:
    exp_dir = Path(__file__).parent.parent / "experiment"
    #exp_setting = exp_dir / "traci_tls"
    exp_setting = exp_dir / "bologna_clean"
    data_dir = exp_setting / "data" / "test1"
    train_data_dir = data_dir  / "train_data"
    test_data_dir = data_dir / "test_data"
    out_dir = data_dir / "out"
    name = f"dimension_{dimension}_n_heads_{n_heads}_n_layer_{n_layer}"
    log_folder_path = out_dir / "Log"
    logger = setup_logger(name, log_folder_path)
    logger.info(f"========== process {dimension}_{n_heads}_{n_layer}  is running! ===========" )
    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)
    
    veh_depart = load_veh_depart("veh_depart", train_data_dir)
    veh_route = load_veh_route("veh_route", train_data_dir)
    logger.info(f"========== finish load route and depart ========")
    # init struc_dict, feat_dict, node_id_dict
    struc_dict, feat_dict, node_id_dict, scalers =  load_graph(train_data_dir)
    #test_struc, test_feat, node_id_dict, scalers =  load_graph(test_data_dir)
    logger.info(f"========= finish load graph =========")
    #model parameters
    n_epochs = 1
    batch_size = 1
    lr = 1e-3
    hid_dim = dimension
    n_heads = n_heads
    changable_feature_names = ['speed','pos_on_lane','occupancy','acceleration']
    graph = Graph(struc_dict, feat_dict)
    hetero_feat_dim = graph.hetero_feat_dim
    timestamps = graph.timestamps.float().tolist()
    
    encoder = HGT(
        in_dim={
            ntype: int(sum(hetero_feat_dim[ntype].values()))
            for ntype in hetero_feat_dim.keys()
        },
        n_ntypes=graph.num_ntypes,
        n_etypes=graph.num_etypes,
        hid_dim=hid_dim,
        n_layers=n_layer,
        n_heads=n_heads,
        activation = nn.ReLU()
    )


    generator = RuleBasedGenerator(
        hetero_feat_dim,
        n_heads * hid_dim,
        {
            ntype: int(sum(hetero_feat_dim[ntype].values()))
            for ntype in hetero_feat_dim.keys()
        },
        activation = nn.ReLU(),
        scalers= scalers
    )
    
    logger.info("========== finish generate generator rule ==========")
    generator.register_rule(post_actions)

    loss_fcn = GraphStateLoss()

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr)
    
    #before = datetime.now()
    for ep in tqdm(range(n_epochs)):
        logger.info(f"--------- current ep is {ep} --------")
        train(timestamps, graph, batch_size, encoder, generator, veh_route, loss_fcn, optimizer, logger)
    
    torch.save(encoder.state_dict(), out_dir)
    torch.save(generator.state_dict(), out_dir)
    
    sim_graph = eval(graph, batch_size, encoder, generator, veh_depart, veh_route, changable_feature_names, hetero_feat_dim)
    #print(sim_graph['veh/0@198.0'][0][('veh','phy/to','lane')][2])
    
    with open(out_dir / f"predicted_graph_{n_layer}_{n_heads}_{dimension}.p", "wb") as f:
        pickle.dump(sim_graph, f)  
    # with open(out_dir / f"node_id_dict_{n_layer}_{n_heads}_{dimension}.p", "wb") as f:
    #     pickle.dump(node_id_dict, f) 

    #after = datetime.now()
    #time_diff.append((after - before).total_seconds())
    
    print("============= Exp has finished! ========")


if __name__ =="__main__":
    # pool = Pool(3)
    hidden_dim_list = [50,100,200]
    n_head_list = [2,4,8]
    n_layer_list = [1,2,3,4]
    for hidden_dim in hidden_dim_list:
        for n_head in n_head_list:
            for n_layer in n_layer_list:
                # if hidden_dim == 50 and n_head == 2 and n_layer in [1,2,3]:
                #     continue
                # pool.apply(run, (hidden_dim, n_head, n_layer,))
                run(hidden_dim, n_head, n_layer)
                game.core.controller.Controller.reset()
                game.core.node.Node.reset()
    # pool.close()
    # pool.join()
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dimension", type=int, default=50)
    # parser.add_argument("--n_head", type=int, default=4)
    # parser.add_argument("--n_layer", type=int, default=2)
    # args = parser.parse_args()
    # run(args.dimension, args.n_head, args.n_layer)