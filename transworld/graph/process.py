import pandas as pd
from collections import defaultdict
import torch
from typing import Dict
import ast

def generate_unique_node_id(node_data: pd.DataFrame) -> Dict[str, int]:
    node_id_dict = {}
    # for node_type in set(node_data["type"]):
    for node_type in node_data["type"].unique():
        # unique_names = list(set(node_data[node_data["type"] == node_type].name))
        unique_names = list((node_data[node_data["type"] == node_type].name).unique())
        node_id_dict_ = {name: unique_names.index(name) for name in unique_names}
        node_id_dict.update(node_id_dict_)
    return node_id_dict


def generate_graph_dict(edge_Data: pd.DataFrame, start_step, end_step) -> Dict:
    g_dict: Dict = defaultdict(list)
    edge_Data = edge_Data.loc[(edge_Data['step'] >= start_step) & (edge_Data['step'] <= end_step)]
    for rel in edge_Data.relation.unique():
        sub_edge = edge_Data[edge_Data["relation"] == rel]
        orig_type, rel_type, dest_type = rel.split("_")
        g_dict[(orig_type, rel_type, dest_type)] = [
            torch.tensor(
                [
                    sub_edge.iloc[i]["from_id"],
                    sub_edge.iloc[i]["to_id"],
                    sub_edge.iloc[i]["step"],
                ]
            )
            for i in range(sub_edge.shape[0])
        ]

    for k, v in g_dict.items():
        tensor_ = torch.cat([t.unsqueeze(1) for t in v], dim=1)
        g_dict[k] = (tensor_[0], tensor_[1], tensor_[2])
    return g_dict


def generate_feat_dict(node_type: str, feat_data: pd.DataFrame, start_step, end_step) -> Dict:
    feat_data = feat_data.loc[(feat_data['step'] >= start_step) & (feat_data['step'] <= end_step)]
    g_feat_dict: Dict = defaultdict(dict)
    node_id_list = feat_data.node_id.unique()
    sub_feat_data = feat_data.drop(["name", "node_id"], axis=1)
    for node_i in node_id_list:
        sub_node = feat_data[feat_data.node_id == node_i]
        
        feat_dict = {
            feat_i: torch.tensor(list([ast.literal_eval(str(x)) for x in sub_node[feat_i]]))
            for feat_i in sub_feat_data.columns
        }
        # feat_dict["step_keep"] = feat_dict["step"]
        feat_dict["time"] = feat_dict.pop("step")
        g_feat_dict[node_i].update(feat_dict)

    return {node_type: g_feat_dict}
