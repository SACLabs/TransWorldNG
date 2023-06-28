from typing import Dict, Union, List
from collections import defaultdict
from graph.process import generate_unique_node_id
import pandas as pd
from pathlib import Path


def load_veh_depart(filename, data_path: Path, training_step: int) -> Dict:
    pass


def pre_actions(veh_depart, sys_time, subgraph):
    """
    Add new vehicles to the system if it is ready to departure.
    return: add node/edge action, for example "add_node(v-h/1)" and  "add_edge(veh/1, phy/to, lane/1)"
    """
    actions = {}
    return actions
