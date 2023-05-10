from pathlib import Path
from graph.load import load_graph
from graph.process import generate_unique_node_id
from rules.pre_process import load_veh_depart, pre_actions
from rules.post_process import load_veh_route, post_actions
import random
import torch.nn as nn
from game.model import HGT, RuleBasedGenerator, GraphStateLoss
from game.data import DataLoader, Dataset
from game.graph import Graph
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
import pandas as pd
from collections import defaultdict
random.seed(3407)

def print_loss(epoch, i, loss):
    with open('loss.csv','a') as loss_log:
        train_writer = csv.writer(loss_log)
        train_writer.writerow([str(epoch), str(i), str(round(loss,4))])

def train(timestamps, graph, batch_size, num_workers, encoder, generator, veh_route, loss_fcn, optimizer, logger, device):
    logger.info("========= start generate dataset =======")
    train_dataset = Dataset(timestamps, device, train_mode=True)
    train_loader = DataLoader(train_dataset, graph.operate, batch_size=batch_size, num_workers=num_workers, drop_last =True)
    logger.info("========== finish generate dataset =======")
    # graph_dicts = {}
    logger.info("========= start training =======")
    loss_list = []
    for i, (cur_graphs, next_graphs) in enumerate(train_loader):  # 这里for的是时间戳
        # 下面这个for循环后面会改成并发操作
        loss = 0.
        for ((_, cur_graph), (seed_node_n_time, next_graph)) in zip(cur_graphs.items(), next_graphs.items()):  
            cur_graph, next_graph = cur_graph.to(device), next_graph.to(device)
            assert (_.split("@")[0]) == seed_node_n_time.split("@")[0], ValueError("Dataloader Error! node_name not equal")
            node_type = seed_node_n_time.split("/")[0]
            # if node_type == 'veh':
            #     continue
            time = float(seed_node_n_time.split("@")[1])
            node_repr = encoder(cur_graph)
            actions, pred_graph = generator([seed_node_n_time], cur_graph, node_repr, veh_route)
            loss = loss + loss_fcn(pred_graph, next_graph)
        loss_list.append((loss.item()) / batch_size)
        #print_loss(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f"------------ loss is {sum(loss_list) / len(loss_list)} ---------")
    logger.info("========== finished training ==========")
    torch.cuda.empty_cache()
    return loss_list

@torch.no_grad()
def eval(graph, batch_size, num_workers, encoder, generator, veh_depart, veh_route, changable_feature_names, hetero_feat_dim, logger, device, training_step, perd_step):
    val_timestamp = [float(i) for i in range(int(training_step),int(training_step)+perd_step+1)]
    #print(val_timestamp)
    val_dataset = Dataset(val_timestamp, device, train_mode=False)
    val_loader = DataLoader(val_dataset, graph.operate, batch_size=batch_size, num_workers=num_workers, drop_last =False)
    logger.info(f"========= start eval ======= batch_size{batch_size}=======")
    for i, cur_graphs in tqdm(enumerate(val_loader)):
        val_result = {}
        for seed_node_n_time, cur_graph in cur_graphs.items():
            cur_graph = cur_graph.to(device)
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
                
    total_graphs = val_loader.collate_tool([max(val_timestamp)+1.], max_step=perd_step)
    for seed_node_n_time, total_graph in total_graphs[0].items():
        struc_dict, feat_dict = dgl_graph_to_graph_dict(total_graph, hetero_feat_dim)
        val_result[seed_node_n_time] = (struc_dict, feat_dict)
    logger.info("========== finished eval ==========")
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

def run(scenario, test_data, training_step, pred_step, hid_dim, n_heads, n_layer, device):
    time_diff = []
    #for test in [1,2,3,4,5]:
    exp_dir = Path(__file__).parent.parent / "experiment"
    
    #exp_setting = exp_dir / "traci_tls"
    #exp_setting = exp_dir / "bologna_clean"
    exp_setting = exp_dir / scenario
    data_dir = exp_setting / "data" / test_data
    train_data_dir = data_dir  / "train_data"
    test_data_dir = data_dir / "test_data"
    out_dir = data_dir / f"out_dim_{hid_dim}_n_heads_{n_heads}_n_layer_{n_layer}_pred_step_{pred_step}"
    name = f"scenario_{scenario}test_data_{test_data}_dim_{hid_dim}_n_heads_{n_heads}_n_layer_{n_layer}"
    log_folder_path = out_dir / "Log"
    logger = setup_logger(name, log_folder_path)
    logger.info(f"========== process {scenario}_{test_data}_{hid_dim}_{n_heads}_{n_layer}_pred_step_{pred_step}  is running! ===========" )
    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)
    
    node_all = pd.read_csv(train_data_dir / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)
    
    veh_depart = load_veh_depart("veh_depart", train_data_dir, training_step)
    veh_route = load_veh_route("veh_route", train_data_dir)
    logger.info(f"========== finish load route and depart ========")
    # init struc_dict, feat_dict, node_id_dict
    
    struc_dict, feat_dict, node_id_dict, scalers =  load_graph(train_data_dir, 0, training_step-1, node_id_dict)
    #test_struc, test_feat, node_id_dict, scalers =  load_graph(test_data_dir)
    logger.info(f"========= finish load graph =========")
    #model parameters
    n_epochs = 100 #200
    batch_size = max(4,training_step//10 - 10) #100
    num_workers = 10 #10
    batch_size = max(1, batch_size * num_workers)
    lr = 5e-4
    hid_dim = hid_dim
    n_heads = n_heads
    changable_feature_names = ['speed','pos_on_lane','occupancy','acceleration']
    graph = Graph(struc_dict, feat_dict)
    hetero_feat_dim = graph.hetero_feat_dim
    timestamps = graph.timestamps.float().tolist()
    
    logger.info(f"========= {n_epochs}_{batch_size}_{num_workers} =========")
    
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
    ).to(device)


    generator = RuleBasedGenerator(
        hetero_feat_dim,
        n_heads * hid_dim,
        {
            ntype: int(sum(hetero_feat_dim[ntype].values()))
            for ntype in hetero_feat_dim.keys()
        },
        activation = nn.ReLU(),
        scalers= scalers,
        output_activation = nn.Sigmoid()
    ).to(device)
    
    logger.info("========== finish generate generator rule ==========")
    generator.register_rule(post_actions)

    loss_fcn = GraphStateLoss().to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr)
    
    # before = datetime.now()
    
    loss_avg = []
    for ep in tqdm(range(n_epochs)):
        logger.info(f"--------- current ep is {ep} --------")
        loss_lst = train(timestamps, graph, batch_size, num_workers, encoder, generator, veh_route, loss_fcn, optimizer, logger, device)
        #loss_dict[f'train_{ep}_loss'] = loss_lst
        loss_avg.append(sum(loss_lst) / len(loss_lst))
    
  
    loss_df = pd.DataFrame(loss_avg)
    # loss_df = pd.DataFrame.from_dict(dict(loss_dict))
    loss_df.to_csv(out_dir / 'loss.csv', index=False)
    
    torch.save(encoder.state_dict(), out_dir / 'encorder.pth')
    torch.save(generator.state_dict(), out_dir / 'generator.pth')
    
    # model_dir = exp_setting / "data" / 'test100' / 'out_dim_100_n_heads_4_n_layer_4_pred_step_10'
    
    # encoder_path = model_dir / 'encorder.pth'
    # generator_path = model_dir / 'generator.pth'
    
    # encoder.load_state_dict(torch.load(encoder_path))
    # generator.load_state_dict(torch.load(generator_path))

    before = datetime.now()
    
    for i in range(10):
        logger.info(f"--------- current is {0+pred_step*(i+1), training_step+pred_step*(i+1)} --------")
        sim_graph = eval(graph, batch_size//num_workers, num_workers, encoder, generator, veh_depart, veh_route, changable_feature_names, hetero_feat_dim, logger, device, training_step+pred_step*(i+1), pred_step)
        #print(sim_graph['veh/0@198.0'][0][('veh','phy/to','lane')][2])
        with open(out_dir / f"predicted_graph_{scenario}_{test_data}_{n_layer}_{n_heads}_{hid_dim}_{i}.p", "wb") as f:
            pickle.dump(sim_graph, f)  
        graph.reset()
        #struc_dict, feat_dict, node_id_dict, scalers =  load_graph(train_data_dir, 0+pred_step*(i+1), training_step+pred_step*(i+1), node_id_dict)
        struc_dict, feat_dict, node_id_dict, scalers =  load_graph(train_data_dir, 0, training_step+pred_step*(i+1), node_id_dict)
        veh_depart = load_veh_depart("veh_depart", train_data_dir, training_step+pred_step*(i+1))
        #logger.info(f"--------- current is {0+pred_step*(i+1), training_step+pred_step*(i+1)} --------")
        graph = Graph(struc_dict, feat_dict)
        #print(0+pred_step*(i+1), training_step+pred_step*(i+1))

    # with open(out_dir / f"node_id_dict_{n_layer}_{n_heads}_{hid_dim}.p", "wb") as f:
    #     pickle.dump(node_id_dict, f) 

    after = datetime.now()
    #time_diff.append((after - before).total_seconds())
    
    logger.info(f"========== time_diff is : {(after - before).total_seconds()} ==========")
    logger.info("========== Exp has finished! ==========")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default='traci_tls')
    parser.add_argument("--test_data", type=str, default='test100')
    parser.add_argument("--training_step", type=int, default=80)
    parser.add_argument("--pred_step", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=100)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if (not torch.cuda.is_available()) or (args.gpu == -1):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda",args.gpu)
    run(args.scenario,args.test_data, args.training_step, args.pred_step, args.hid_dim, args.n_head, args.n_layer, device)