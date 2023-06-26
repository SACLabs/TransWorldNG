from typing import Dict, Union, List
from collections import defaultdict
from graph.process import generate_unique_node_id
import pandas as pd
from pathlib import Path
import csv

def load_veh_route(filename, data_path: Path) -> Dict:
    pass


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


def get_current_phase_state(plan, current_time):
    phase_start_time = 0
    cycle_time = sum(plan[0][3])
    current_time_in_cycle = current_time % cycle_time
    for inbound, outbound, state, phaseduration in plan:
        for i in range(len(phaseduration)):
            phase_end_time = phase_start_time + phaseduration[i]
            if current_time_in_cycle < phase_end_time:
                return i+1, state[i]
            phase_start_time = phase_end_time % cycle_time
    return None, None


def post_actions(
    node_names: List[str], struc_dict: Dict, feat_dict: Dict, veh_route: Dict,
) -> Dict:  # return action: ["node_name","add_edge(veh1,on,lane1)"]
    struc_actions = defaultdict(list)

    inbound_lst = struc_dict[("lane", "phy/to", "lane")][0]
    outbound_lst = struc_dict[("lane", "phy/to", "lane")][1]
    time_lst = struc_dict[("lane", "phy/to", "lane")][2]
    action_time = node_names[0].split('@')[-1] # TODO this code only support full graph inference
    action = []
    aggr_edge_list = defaultdict(float)
    
    for inbound, outbound, time in zip(inbound_lst, outbound_lst, time_lst):
        inbound, outbound, time  = int(inbound), int(outbound), int(time)
        if aggr_edge_list.get((inbound, outbound), None) is None:
            aggr_edge_list[(inbound, outbound)] = {'time':time}
        else:
            new_value_dict = aggr_edge_list[(inbound, outbound)] if aggr_edge_list[(inbound, outbound)]['time'] > time else {'time':time}
            aggr_edge_list[(inbound, outbound)] = new_value_dict

    print(aggr_edge_list)


    plan_df = pd.read_csv("/mnt/workspace/wangding/Desktop/TransWorldNG/experiment/hangzhou/data/run1/train_data/tlc_plans_id.csv")

    plan = []
    for _, row in plan_df.iterrows():
        inbound = int(row['inbound'])
        outbound = int(row['outbound'])
        state = row['state']
        duration = eval(row['duration'])
        plan.append((inbound, outbound, state, duration))

    print(plan)


    cur_time = node_names[0].split('@')[-1]

    cur_phase, cur_state = get_current_phase_state(plan, cur_time)


    print(cur_time, cur_phase, cur_state)

    signal_states = {}  
    for inbound, outbound, state, duration in plan:
        for i in range(len(duration)):
            start_time = sum(duration[:i])  # 当前状态的起始时间
            end_time = start_time + duration[i]  # 当前状态的结束时间
            for t in range(start_time, end_time):
                # 记录当前时间以及在当前时间下的信号状态
                time_in_state = t - start_time
                signal_states[(inbound, outbound, time_in_state)] = state[i]

    print(signal_states)


    for inbound, outbound, state, duration in plan:
        state = signal_states[(inbound, outbound, 0)]
        if state == "G":
            # 添加边
            if (inbound, outbound) not in aggr_edge_list:
                aggr_edge_list[(inbound, outbound)] = {"time": 0, "outbound": outbound}
                action.append(
                    "add_edge(veh/"
                    + str(veh_id)
                    + ",phy/to,"
                    + "lane/"
                    + str(outbound)
                    + ")"
                )
        else:
            # 删除边
            if (inbound, outbound) in aggr_edge_list:
                del aggr_edge_list[(inbound, outbound)]
                action.append(
                    "remove_edge(veh/"
                    + str(veh_id)
                    + ",phy/to,"
                    + "lane/"
                    + str(outbound)
                    + ")"
                )
        
    if action != []:
        struc_actions.update({'veh/'+str(veh_id)+'@'+str(action_time): action})
    return struc_actions


    # inbound_lst = struc_dict[("lane", "phy/to", "lane")][0]
    # outbound_lst = struc_dict[("lane", "phy/to", "lane")][1]
    # time_lst = struc_dict[("lane", "phy/to", "lane")][2]
    # action_time = node_names[0].split('@')[-1] # TODO this code only support full graph inference
    # action = []
    # aggr_edge_list = defaultdict(float)
    
    # for inbound, outbound, time in zip(inbound_lst, outbound_lst, time_lst):
    #     inbound, outbound, time  = int(inbound), int(outbound), int(time)
    #     if aggr_edge_list.get((inbound, outbound), None) is None:
    #         aggr_edge_list[(inbound, outbound)] = {'time':time}
    #     else:
    #         new_value_dict = aggr_edge_list[(inbound, outbound)] if aggr_edge_list[(inbound, outbound)]['time'] > time else {'time':time}
    #         aggr_edge_list[(inbound, outbound)] = new_value_dict


    # for veh_id in list(aggr_edge_list.keys()):
    #     lane_id = aggr_edge_list[veh_id]['lane_node']
    #     current_lane_len = abs(feat_dict["lane"][lane_id]["length"])
    #     pos_on_lane = abs(feat_dict["veh"][veh_id]["pos_on_lane"])

    #     if pos_on_lane / current_lane_len > min_dis:
    #         next_lane = get_veh_next_lane(veh_id, veh_route, lane_id)
    #         tlc_state = feat_dict["veh"][veh_id]["tlc_state"]
    #         if next_lane is None:  # This vehicle has reached the destination
    #             action.append("delete_node(veh/" + str(veh_id)+ ")")
    #         # elif (next_lane is not None) and (
    #         #     tlc_state >= 0
    #         # ):  # This vehicle will move to it's next route if it's upcoming tlc state is either green(1) or yellow(0)
    #         elif next_lane is not None:
    #             action.append(
    #                 "add_edge(veh/"
    #                 + str(veh_id)
    #                 + ",phy/to,"
    #                 + "lane/"
    #                 + str(next_lane)
    #                 + ")"
    #             )
    #             # action.append(
    #             #     "delete_edge(veh/"
    #             #     + str(veh_id)
    #             #     + ",phy/to,"
    #             #     + "lane/"
    #             #     + str(lane_id)
    #             #     + ")"
    #             #)

    # return struc_actions
    
    
    
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
    pass
