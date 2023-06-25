from __future__ import absolute_import
from __future__ import print_function
from typing import List, Optional, Union, Tuple
from functools import singledispatch
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import csv
import ast
import itertools
from tqdm import tqdm

try:
    tools = os.path.join(os.environ.get("SUMO_HOME"), "tools")
    sys.path.append(tools)
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME'")


import sumolib
import traci


# def get_veh_depart(target_step, data_path: Path, save_dir: Path) -> pd.DataFrame:
#     veh_depart = []
#     veh_depart.append(["name", "depart", "entry"])
#     for veh in sumolib.output.parse(data_path, "vehicle"):
#         if int(veh.depart) < target_step - 1:
#             route_id = veh.route
#             route_list = [
#                 edge
#                 for edge in sumolib.output.parse(data_path, "route")
#                 if edge.id == route_id
#             ][0].edges.split()
#             entry = route_list[0]
#             veh_depart.append([veh.id, veh.depart, entry])
#     save_data([("veh_depart", veh_depart)], save_dir)
#     return veh_depart
def get_veh_depart(target_step, data_path: Path, save_dir: Path) -> pd.DataFrame:
    veh_depart = []
    veh_depart.append(["name", "depart", "entry"])
    for veh in tqdm(sumolib.output.parse(data_path, "vehicle")):
        if int(veh.depart) < target_step :
            route_id = veh.route
            route_list = [
                edge
                for edge in sumolib.output.parse(data_path, "route")
                if edge.id == route_id
            ]
            if route_list != []:
                route_list1 = route_list[0].edges.split()
                entry = route_list1[0]
                veh_depart.append([veh.id, veh.depart, entry])
    save_data([("veh_depart", veh_depart)], save_dir)
    return veh_depart

def get_veh_route(target_step, data_path: Path, save_dir: Path) -> pd.DataFrame:
    veh_route = []
    veh_route.append(["name", "route"])
    for veh in tqdm(sumolib.output.parse(data_path, "vehicle")):
        if int(veh.depart) < target_step :
            route_id = veh.route
            route_list = [
                edge
                for edge in sumolib.output.parse(data_path, "route")
                if edge.id == route_id
            ]
            if route_list != []:
                route_list1 = route_list[0].edges.split()
                veh_route.append([veh.id, route_list1])
    save_data([("veh_route", veh_route)], save_dir)
    return veh_route


def get_follower_veh_id(veh: str) -> Optional[str]:
    if traci.vehicle.getFollower(veh) is not None:
        veh_id = traci.vehicle.getFollower(veh)[0]
        return veh_id
    else:
        return None


def get_follower_veh_dis(veh: str) -> Optional[str]:
    if traci.vehicle.getFollower(veh) is not None:
        veh_dis = traci.vehicle.getFollower(veh)[1]
        return veh_dis
    else:
        return None

def get_tlc_state(veh: str) -> Optional[int]:
    next_tls = traci.vehicle.getNextTLS(veh)
    if len(next_tls) > 0 and next_tls[0][-1] == "G":
        return 1
    if len(next_tls) > 0 and next_tls[0][-1] == "r":
        return -1
    elif len(next_tls) > 0 and next_tls[0][-1] == "s":
        return 0
    else:
        return 0



def get_node_data(step: int, type: str, sources: List, operations: List) -> List:
    return_nodes = []
    return_feats = []
    for element in sources:
        return_nodes.append([step, element, type])
        operation_result = [
            operator(element)
            for operator in operations
            if operator(element) is not None
        ] 
        if type == "lane":
            operation_result[-1] = tuple(itertools.chain(*operation_result[-1]))

        # if type == "tlc":
        #     print(traci.trafficlight.getControlledLinks(element))
        return_feats.append([step, element] + operation_result )
    return return_nodes, return_feats


def get_edge_data(step: int, relation: str, sources: List, operations: List) -> List:
    return_edges = []
    for element in sources:
        if relation == "lane_to_lane":
            
            dest = [
                operator(element)[0][0]
                for operator in operations
                if (len(operator(element)) > 0)
            ]
            
            
            if dest!= []:
                if type(dest) == list:
                    dest = dest[0]
                return_edges.append([step, element, dest, "lane_phy/to_lane"])
                return_edges.append([step, dest, element, "lane_phy/from_lane"])

            # Check if the current lane is controlled by a traffic light
            # lane_controlled = False
            # tl_ids = traci.trafficlight.getIDList()
            # for tl_id in tl_ids:
            #     tl_lanes = traci.trafficlight.getControlledLanes(tl_id)
            #     if element in tl_lanes:
            #         lane_controlled = True
            #         break
            
            # # If the current lane is not controlled by a traffic light, add it to the list of edges
            # if not lane_controlled:
            #     dest = [
            #         operator(element)[0][0]
            #         for operator in operations
            #         if (len(operator(element)) > 0)
            #     ]
            #     if dest!= []:
            #         if type(dest) == list:
            #             dest = dest[0]
            #         return_edges.append([step, element, dest, "lane_phy/to_lane"])
            #         return_edges.append([step, dest, element, "lane_phy/from_lane"])
        
        elif relation == "lane_belongs_road":
            dest = [
                operator(element)
                for operator in operations
                if len(operator(element)) > 0
            ][0]
            return_edges.append([step, element, dest, "lane_phy/to_road"])
            return_edges.append([step, dest, element, "road_phy/to_lane"])
        
        elif relation == "tlc_to_lane":
            lanes = [
                operator(element)
                for operator in operations
                if len(operator(element)) > 0
            ][0]

            for lane in lanes:
                return_edges.append([step, element, lane, "tlc_phy/to_lane"])
                return_edges.append([step, lane, element, "lane_phy/to_tlc"])

        elif relation == "veh_on_lane":
            vehs = [
                operator(element)
                for operator in operations
                if len(operator(element)) > 0
            ]
            #print(vehs)
            if vehs != []:
                for veh in vehs[0]:
                    return_edges.append([step, veh, element, "veh_phy/to_lane"])
                    return_edges.append([step, element, veh, "lane_phy/to_veh"])

        elif relation == "veh_follow_veh":
            followers = [
                operator(element)
                for operator in operations
                if len(operator(element)) > 0
            ]
            for follower in followers:
                return_edges.append([step, follower, element, "veh_phy/behind_veh"])
                return_edges.append([step, element, follower, "veh_phy/ahead_veh"])

        elif relation == "veh_to_tlc":
            next_tls = [
                operator(element) for operator in operations
            ]  # if ((operator(element) != []) and (len(operator(element)[0]) > 0))]
            if len(next_tls[0]) > 0:
                next_tls_id = next_tls[0][0][0]
                return_edges.append([step, element, next_tls_id, "veh_phy/to_tlc"])
                return_edges.append([step, next_tls_id, element, "tlc_phy/to_veh"])
        else:
            raise ValueError(f"Relation {relation} unknown.")

    return return_edges



@singledispatch
def save_data(data_entity: Union[List[Tuple], Tuple], data_dir: str):
    raise NotImplementedError


@save_data.register(list)
def _(data_entity, data_dir):
    for tuple_ in data_entity:
        save_data(tuple_, data_dir)


@save_data.register(tuple)
def _(data_entity, data_dir):
    filename, data = data_entity
    with open(Path(data_dir) / (filename + ".csv"), "w", newline="") as graph_file:
        writer = csv.writer(graph_file, delimiter=",")
        for line in data:
            writer.writerow(line)


def get_tlc_plan(tlc_id):
    # num_phase = len(traci.trafficlight.getAllProgramLogics(tlc_id)[0].phases)
    program_logic = traci.trafficlight.getAllProgramLogics(tlc_id)[0]
    num_phase = len(program_logic.phases)
    controlled_links = traci.trafficlight.getControlledLinks(tlc_id)
    inbound_lst = [link[0][0] for link in controlled_links]
    outbound_lst = [link[0][1] for link in controlled_links]
    state_lst = [
        program_logic.phases[i].state
        for i in range(num_phase)
    ]
    # Reorganize the phase states
    state_tuples = [[char for char in state] for state in state_lst]
    #print('state_tuples',state_tuples)
    state_lst = ["".join(chars) for chars in zip(*state_tuples)]

    tlc_plan = []
    for i in range(num_phase):
        phase_duration_lst = [program_logic.phases[i].duration for i in range(num_phase)]
        tlc_plan.append([inbound_lst[i], outbound_lst[i], state_lst[i], phase_duration_lst])

    #tlc_plan = [inbound_lst, outbound_lst, state_lst, phase_duration_lst]

    # Transpose the tlc_plan list
    #tlc_plan = list(map(list, zip(*tlc_plan)))
    #print(tlc_plan)
    
    return tlc_plan


def run_sumo(target_step: int, train_data_dir: Path, test_data_dir: Path ) -> None:

    """initialize the graph data list"""
    g_node = []
    g_edge = []
    feat_veh = []
    feat_lane = []
    feat_tlc = []
    feat_road = []
    g_node.append(["step", "name", "type"])
    g_edge.append(["step", "from", "to", "relation"])
    feat_tlc.append(["step", "name", "program","phase"])
    feat_road.append(["step", "name", "lane_num"])
    feat_veh.append(
        [
            "step",
            "name",
            "pos_on_lane",
            "length",
            "vmax",
            "speed",
            "acceleration",
            "tlc_state",
            "coordinate",
        ]
    )
    
    feat_lane.append(["step", "name", "length","vehicles", "occupancy","speed","shape"])
    
    

    tlc_plans = []
    tlc_plans.append(["inbound", "outbound", "state","duration"])
    for tlc_id in traci.trafficlight.getIDList():
        tlc_plans.extend(get_tlc_plan(tlc_id))

    # time_diffs = []
    # num_cars = []
    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
    #while step <121:
        
        traci.simulationStep()
        
        step += 1
        
        """Get node and node feature"""
        node, feat = get_node_data(
            step, "road", traci.edge.getIDList(), [traci.edge.getLaneNumber]
        )

        g_node.extend(node)
        feat_road.extend(feat)

        node, feat = get_node_data(
            step,
            "tlc",
            traci.trafficlight.getIDList(),
            [
                traci.trafficlight.getProgram,
                traci.trafficlight.getPhase,
            ],
        )
        g_node.extend(node)
        feat_tlc.extend(feat)

        node, feat = get_node_data(
            step,
            "lane",
            traci.lane.getIDList(),
            [
                traci.lane.getLength,
                traci.lane.getLastStepVehicleNumber,
                traci.lane.getLastStepOccupancy,
                traci.lane.getLastStepMeanSpeed,
                traci.lane.getShape,
            ],
        )

        g_node.extend(node)
        feat_lane.extend(feat)

        node, feat = get_node_data(
            step,
            "veh",
            traci.vehicle.getIDList(),
            [
                traci.vehicle.getLanePosition,
                traci.vehicle.getLength,
                traci.vehicle.getMaxSpeed,
                traci.vehicle.getSpeed,
                traci.vehicle.getAcceleration,
                get_tlc_state,
                traci.vehicle.getPosition,
            ],
        )

        g_node.extend(node)
        feat_veh.extend(feat)

        """Get edge"""
        edge_data = get_edge_data(
            step,
            "lane_belongs_road",
            traci.lane.getIDList(),
            [traci.lane.getEdgeID],
        )
        g_edge.extend(edge_data)
        
        edge_data = get_edge_data(
            step,
            "veh_on_lane",
            traci.lane.getIDList(),
            [traci.lane.getLastStepVehicleIDs],
        )
        
        # if step in [12,13]:
        #     print(edge_data, traci.lane.getIDList(), )
        #     print([traci.lane.getLastStepVehicleIDs(i) for i in traci.lane.getIDList()])
        g_edge.extend(edge_data)

        edge_data = get_edge_data(
            step,
            "lane_to_lane",
            traci.lane.getIDList(),
            [traci.lane.getLinks],
        )
        g_edge.extend(edge_data)

        edge_data = get_edge_data(
            step,
            "veh_to_tlc",
            traci.vehicle.getIDList(),
            [traci.vehicle.getNextTLS],
        )
        g_edge.extend(edge_data)

        edge_data = get_edge_data(
            step, "veh_follow_veh", traci.vehicle.getIDList(), [get_follower_veh_id]
        )
        g_edge.extend(edge_data)

        edge_data = get_edge_data(
            step,
            "tlc_to_lane",
            traci.trafficlight.getIDList(),
            [traci.trafficlight.getControlledLanes],
        )
        g_edge.extend(edge_data)
    
        if step == target_step:
            """Train dataset: save graph and features to csv"""
            save_data(
                [
                    ("node", g_node),
                    ("edge", g_edge),
                    ("feat_veh", feat_veh),
                    ("feat_lane", feat_lane),
                    ("feat_tlc", feat_tlc),
                    ("feat_road", feat_road),
                    ("tlc_plans", tlc_plans),
                ],
                train_data_dir,
            )
            print("step",step, ":Train dataset saved!")
        if step == target_step+20:
            """Test dataset: save graph and features to csv"""
            save_data(
                [
                    ("node", g_node),
                    ("edge", g_edge),
                    ("feat_veh", feat_veh),
                    ("feat_lane", feat_lane),
                    ("feat_tlc", feat_tlc),
                    ("feat_road", feat_road),
                    ("tlc_plans", tlc_plans),
                ],
                test_data_dir,
            )
            print("step",step, ":Test dataset saved!")
    save_data([("node_all", g_node)], train_data_dir)
    save_data([("node_all", g_node)], test_data_dir)
    
    traci.close()
    sys.stdout.flush()
