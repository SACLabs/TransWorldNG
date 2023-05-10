from pathlib import Path
from graph.load import load_graph
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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

random.seed(111)


def train():
    train_dataset = Dataset(timestamps, train_mode=True)
    train_loader = DataLoader(train_dataset, graph.operate, batch_size=batch_size, num_workers=0)
    # graph_dicts = {}
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval():
    val_timestamp = [float(i) for i in range(100,199)]
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
                
    total_graphs = val_loader.collate_tool([max(val_timestamp)+1.], max_step=99)
    for seed_node_n_time, total_graph in total_graphs[0].items():
        struc_dict, feat_dict = dgl_graph_to_graph_dict(total_graph, hetero_feat_dim)
        val_result[seed_node_n_time] = (struc_dict, feat_dict)
    return val_result      


if __name__ == "__main__":
    time_diff = []
    #for test in [1,2,3,4,5]:
    exp_dir = Path(__file__).parent.parent / "experiment"
    exp_setting = exp_dir / "traci_tls"
    data_dir = exp_setting / "data" / "test1"
    train_data_dir = data_dir  / "train_data"
    test_data_dir = data_dir / "test_data"
    out_dir = data_dir / "out"
    
    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)
    
    veh_depart = load_veh_depart("veh_depart", train_data_dir)
    veh_route = load_veh_route("veh_route", train_data_dir)

    # init struc_dict, feat_dict, node_id_dict
    struc_dict, feat_dict, node_id_dict, scalers =  load_graph(train_data_dir)
    #test_struc, test_feat, node_id_dict, scalers =  load_graph(test_data_dir)

    #model parameters
    n_epochs = 10
    batch_size = 1
    lr = 1e-5
    hid_dim = 20
    n_heads = 2
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
        n_layers=2,
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
        activation = nn.ReLU()
    )
    generator.register_rule(post_actions)

    loss_fcn = GraphStateLoss()

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr)
    
    #before = datetime.now()
    for _ in range(n_epochs):
        train()
    sim_graph = eval()
    #print(sim_graph['veh/0@198.0'][0][('veh','phy/to','lane')][2])
    
    
    
    for key in sim_graph.keys():
        feat_dict = sim_graph[key][1]
        for type in feat_dict.keys():
            feat_type_dict = feat_dict[type]
            for ids in feat_type_dict.keys():
                feat_type_id_dict = feat_type_dict[ids]
                for feat_i in feat_type_id_dict.keys():
                    feat_value = feat_type_id_dict[feat_i].reshape(-1,1)
                    sim_graph[key][type][ids][feat_i] = torch.tensor(scaler.inverse_transform(feat_value).squeeze())
    
    

    with open(out_dir / "predicted_graph.p", "wb") as f:
        pickle.dump(sim_graph, f)  
    with open(out_dir / "node_id_dict.p", "wb") as f:
        pickle.dump(node_id_dict, f) 

    #after = datetime.now()
    #time_diff.append((after - before).total_seconds())
    
    print("============= Exp has finished! ========")
    #print("============= exp {} has finished! ========".format(i))
#print(time_diff)










    # with open(data_dir / "test_struc_dict.p", "wb") as g:
    #     pickle.dump(struc_dict,g)
    # with open(data_dir / "test_feat_dict.p", "wb") as f:
    #     pickle.dump(feat_dict, f)  
    
    
    # # 模型参数
    # n_epochs = 1
    # batch_size = 3
    # lr = 1e-3
    # hid_dim = 20
    # n_layers= 2
    # n_heads = 2

    # graph = Graph(struc_dict, feat_dict)
    # feat_dim = graph.hetero_feat_dim
    # veh_depart = load_veh_depart("veh_depart", data_dir)
    # veh_route = load_veh_route("veh_route", data_dir)
    # timestamps = graph.timestamps.float().tolist()
    # train_dataset = Dataset(timestamps, train_mode=True)
    # train_loader = DataLoader(
    # train_dataset, graph.operate, batch_size=batch_size, num_workers=0
    # )
    
    # model = HGT(
    #     #in_dim=graph.in_dim, #{"veh": 6, "lane": 4, "tlc": 2}
    #     in_dim={ntype: int(sum(feat_dim[ntype].values())) for ntype in feat_dim.keys()},
    #     n_ntypes=graph.num_ntypes,
    #     n_etypes=graph.num_etypes,
    #     hid_dim = hid_dim,
    #     n_layers= n_layers,
    #     n_heads = n_heads,
    # )
    
    # with open(data_dir / "graph_dict.pkl", "rb") as g:
    #     model_graph = pickle.load(g)
    # for node_name in model_graph.keys():
    #     struc_dict, feat_dict = model_graph[node_name]
    #     actions_post = post_actions([node_name], struc_dict, feat_dict, veh_route)
                
    # # for step, (cur_subgraphs, next_subgraphs) in enumerate(train_loader): # 这里for时间
    # #     for cur_subgraph, next_subgraph in zip(cur_subgraphs, next_subgraphs):
    # #         # 1. run rule.preprocess
    # #         actions_pre = pre_actions(veh_depart, step, cur_subgraph)
    # #         #cur_subgraph.update(actions)
    # #         # 2 调用 model
    # #         #model_graph = model(cur_subgraph)   
    # #         for node_name in model_graph.keys():
    # #             struc_dict, feat_dict = model_graph[node_name]
    # #             # node_names = ['veh/2@15.0','veh/0@15.0', 'lane/2@15.0', 'lane/11@15.0']
            
    # #         # 3. run rule.post_process
    # #         actions_post = post_actions([node_names], struc_dict, feat_dict, veh_route)
    # #         #cur_subgraph.update(actions)
    # #         #loss = mseloss(cur_subgraph, next_subgraph)
    # #         #loss.backward()
    


