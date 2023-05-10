from pathlib import Path
from graph.load import load_graph
from rules.pre_process import load_veh_depart, pre_actions
from rules.post_process import post_actions
import pickle
import random
from game.model import HomoToHeteroLinear, HeteroToHomoLinear, HGT
import game

random.seed(111)

if __name__ == "__main__":

    exp_dir = Path(__file__).parent.parent / "experiment"
    exp_setting = exp_dir / "traci_tls"
    data_dir = exp_setting / "data" / "micro"


    # init struc_dict, feat_dict, node_id_dict, run 1000
    sim_time = 1000
    step = 0
    struc_dict, feat_dict, node_id_dict =  load_graph(data_dir)
    graph = game.graph.graph(struc_dict, feat_dict)
    veh_depart = load_veh_depart("veh_depart", data_dir)
    real_struc_dict, real_dict, node_id_dict = load_graph(data_dir)
    test_model = HGT(
        in_dim=graph.in_dim, #{"veh": 6, "lane": 4, "tlc": 2}
        n_ntypes=graph.num_ntypes,
        n_etypes=graph.num_etypes,
        hid_dim = 20,
        n_layers= 2,
        n_heads=2,
    )
    History_list = []
    
    while step < sim_time:
        for subgraph in graph:
            # 1. run rule.preprocess
            actions = pre_actions(veh_depart, step, subgraph)
            subgraph.update(actions)
            # 2 调用model
            model_graph = test_model(subgraph)    
            # 3. run rule.post_process
            actions = post_actions(model_graph.struc_dict, model_graph.feat_dict)
            subgraph.update(actions)
            subgraph.real_update()
            real_graph = game.graph(real_struc_dict[step][name], real_feat_dict[step][name])
            metric = compare(real_graph, subgraph)
            History_list.append(subgraph.sturc_dict, model_graph.feat_dict)
        step+=1
    

