from typing import Dict, Union, List
from collections import defaultdict
from graph.process import generate_unique_node_id
import pandas as pd
from pathlib import Path


def load_veh_route(filename, data_path: Path) -> Dict:
    node_all = pd.read_csv(data_path / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)
    data_file = pd.read_csv(data_path / (filename + ".csv"))
    data_file["veh_id"] = [node_id_dict[str(i)] for i in data_file["name"]]
    rou_id = []
    for rou in data_file["route"]:
        rou = [x for x in rou if x not in ["[", "'", "]", " "]]
        rou = "".join(rou).split(",")
        rou_id.append([node_id_dict[str(i)] for i in rou])
    data_file["route"] = rou_id
    veh_route = data_file.set_index("veh_id").T.to_dict()
    return veh_route


def get_veh_current_lane(struc_dict: Dict) -> int:
    # if ("veh", "phy/to", "lane") in struc_dict.keys():
    lane_id = struc_dict[("veh", "phy/to", "lane")][1]
    return int(lane_id[-1])


def get_veh_next_lane(
    veh_id: int, veh_route: dict, cur_lane_id: int
) -> Union[str, None]:
    route_lst = veh_route[veh_id]["route"]
    cur_lane_idx = route_lst.index(cur_lane_id)
    if cur_lane_idx < len(route_lst) - 1:
        next_lane = route_lst[cur_lane_idx + 1]
        return next_lane
    else:
        return None

def post_actions(
    node_names: List[str], struc_dict: Dict, feat_dict: Dict, veh_route: Dict
) -> Dict:  # return action: ["node_name","add_edge(veh1,on,lane1)"]
    struc_actions = defaultdict(list)
    return struc_actions
    
    
    
    # for node_name in node_names:
    #     action = []
    #     min_dis = 10
    #     node_type, id_step = node_name.split("/")
    #     node_id = int(id_step.split("@")[0])
    #     if node_type == "veh" :
    #         """
    #         Change lane action when approaching the end of lane.
    #         return: move to next lane if availiable, wait if tlc is red, remove node if reached destination
    #         """
    #         min_dis = 10  # minimum distance for decision when approaching the end of lane
            
            
            
    #         current_lane = get_veh_current_lane(struc_dict)
    #         current_lane_len = abs(feat_dict["lane"][current_lane]["length"])
    #         pos_on_lane = abs(feat_dict["veh"][node_id]["pos_on_lane"])
    #         if current_lane_len - pos_on_lane < min_dis:
    #             next_lane = get_veh_next_lane(node_id, veh_route, current_lane)
    #             tlc_state = feat_dict["veh"][node_id]["tlc_state"]
    #             if next_lane is None:  # This vehicle has reached the destination
    #                 action.append("delete_node(veh/" + str(node_id)+ ")")
    #             elif (next_lane is not None) and (
    #                 tlc_state >= 0
    #             ):  # This vehicle will move to it's next route if it's upcoming tlc state is either green(1) or yellow(0)
    #                 action.append(
    #                     "add_edge(veh/"
    #                     + str(node_id)
    #                     + ",phy/to,"
    #                     + "lane/"
    #                     + str(next_lane)
    #                     + ")"
    #                 )
    #                 action.append(
    #                     "delete_edge(veh/"
    #                     + str(node_id)
    #                     + ",phy/to,"
    #                     + "lane/"
    #                     + str(current_lane)
    #                     + ")"
    #                 )
    #     if action != []:
    #         struc_actions.update({node_name: action})
    # return struc_actions


def get_feat_actions(
    node_names: List[str], struc_dict: Dict, feat_dict: Dict, veh_od: Dict
) -> Dict:
    struc_actions = defaultdict(list)
    return struc_actions
