from typing import Dict, Union, List
from collections import defaultdict
from graph.process import generate_unique_node_id
import pandas as pd
from pathlib import Path


def load_veh_depart(filename, data_path: Path, training_step: int) -> Dict:
    node_all = pd.read_csv(data_path / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)
    data_file = pd.read_csv(data_path / (filename + ".csv"))
    #print(training_step)
    data_file = data_file[data_file['depart']<training_step]
    data_file["veh_id"] = [node_id_dict[str(i)] for i in data_file["name"]]
    data_file["lane_id"] = [node_id_dict[str(i)] for i in data_file["entry"]]
    veh_depart = data_file.set_index("depart").T.to_dict()
    return veh_depart


def pre_actions(veh_depart, sys_time, subgraph):
    """
    Add new vehicles to the system if it is ready to departure.
    return: add node/edge action, for example "add_node(v-h/1)" and  "add_edge(veh/1, phy/to, lane/1)"
    """
    #print(veh_depart)
    actions = {}
    for depart in veh_depart.keys():
        #print('veh',depart,sys_time)
        action = []
        if depart == sys_time:
            veh_id = veh_depart[depart]["veh_id"]
            entry = veh_depart[depart]["lane_id"]
            action.append(
                    "add_edge(veh/"
                    + str(veh_id)
                    + ",phy/to,"
                    + "lane/"
                    + str(entry)
                    + ")")
            #print('veh',veh_id,depart,sys_time, action)
        #print('veh',depart,sys_time,action)
          
        if action != []:
            node_name = "veh/" + str(veh_id) + "@" + str(float(sys_time))
            actions.update({node_name: action})

        #print(actions)
    return actions
