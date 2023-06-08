from typing import (
    Any,
    Dict,
    Callable,
    Iterable,
    TypeVar,
    Generic,
    Sequence,
    List,
    Optional,
    Union,
    Optional,
    Tuple,
)
from functools import partial, reduce
from copy import deepcopy
from collections import defaultdict, OrderedDict, ChainMap
import dgl
from dgl import DGLGraph
import torch
from torch.nn import functional as F
from torch import Tensor


def game_to_dgl(
    batched_structures: List,
    batched_features: Dict,
    timestamp: str,
    full_graph: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[DGLGraph]]:
    batched_dgl_graph = OrderedDict()
    if not full_graph:
        for structure in batched_structures:
            if list(structure.values())[0] == {}:
                # 如果找不到这个节点，就不返回
                continue
            seed_node_name = list(structure.keys())[0]
            dgl_graph = structure_to_dgl_graph(structure)
            dgl_graph = dgl.compact_graphs(dgl_graph)
            dgl_graph.ndata["ID"] = dgl_graph.ndata["_ID"]
            del dgl_graph.ndata["_ID"]
            node_feat_dict = feature_to_dgl_feature(dgl_graph, batched_features)

            dgl_graph = dgl_graph_join_feature(dgl_graph, node_feat_dict)
            dgl_graph = attach_node_state(dgl_graph)
            batched_dgl_graph[seed_node_name + "@" + str(timestamp)] = dgl_graph
    else:
        structure = ChainMap(*batched_structures)
        seed_node_name = "full/0"
        dgl_graph = structure_to_dgl_graph(structure)
        dgl_graph = dgl.compact_graphs(dgl_graph)
        dgl_graph.ndata["ID"] = dgl_graph.ndata["_ID"]
        del dgl_graph.ndata["_ID"]
        node_feat_dict = feature_to_dgl_feature(dgl_graph, batched_features)

        dgl_graph = dgl_graph_join_feature(dgl_graph, node_feat_dict)
        dgl_graph = attach_node_state(dgl_graph)
        batched_dgl_graph[seed_node_name + "@" + str(timestamp)] = dgl_graph
    return batched_dgl_graph


def attach_node_state(subgraph: DGLGraph):
    for ntype in subgraph.ntypes:
        feat_names = subgraph.nodes[ntype].data.keys()
        subgraph.nodes[ntype].data["state"] = torch.cat(
            [
                subgraph.nodes[ntype].data[feat_name]
                for feat_name in feat_names
                if "ID" not in feat_name
            ],
            dim=2,
        )
    return subgraph


def update_edge_and_time_tensor_(
    dst_list: List,
    dgl_structure_dict: Dict,
    dgl_edge_time_dict: Dict,
    src_type: str,
    src_id: int,
):
    """push edge information to dgl_structure_dict and dgl_edge_time_dict, from dst_list

    Args:
        dgl_structure_dict (_type_): _description_
        dgl_edge_time_dict (_type_): _description_
        src_type (_type_): _description_
        src_id (_type_): _description_
        dst_list (_type_): _description_
    """
    dst_name, edges = dst_list
    dst_type, dst_id = convert_name_to_type_and_id(dst_name)
    dst_id = torch.tensor(int(dst_id))
    for etype, edge_time in edges:
        dgl_structure_dict[(src_type, etype, dst_type)] = torch.cat(
            [
                dgl_structure_dict[(src_type, etype, dst_type)],
                torch.tensor((src_id, dst_id)).unsqueeze(1),
            ],
            dim=1,
        ).long()
        dgl_edge_time_dict[(src_type, etype, dst_type)] = torch.cat(
            [
                dgl_edge_time_dict[(src_type, etype, dst_type)],
                torch.tensor((edge_time,)),
            ],
            dim=0,
        )


def structure_to_dgl_graph(structure: Dict) -> DGLGraph:
    dgl_structure_dict: Dict = defaultdict(Tensor)
    dgl_edge_time_dict: Dict = defaultdict(Tensor)
    for seed_node in list(structure.keys()):
        src_type, src_id = convert_name_to_type_and_id(seed_node)
        src_id = torch.tensor(int(src_id))
        update_dicts_ = partial(
            update_edge_and_time_tensor_,
            dgl_structure_dict=dgl_structure_dict,
            dgl_edge_time_dict=dgl_edge_time_dict,
            src_type=src_type,
            src_id=src_id,
        )
        list(map(update_dicts_, list(structure[seed_node].items())))
    for key, values in dgl_structure_dict.items():
        dgl_structure_dict[key] = tuple(values)
    dgl_graph = dgl.heterograph(dgl_structure_dict)
    dgl_graph.edata["time"] = (
        dgl_edge_time_dict
        if len(dgl_graph.etypes) > 1
        else list(dgl_edge_time_dict.values())[0]
    )
    return dgl_graph


def feature_to_dgl_feature(dgl_graph: DGLGraph, batched_features: Dict) -> Dict:
    node_feat_dict: Dict = defaultdict(lambda: defaultdict(list))
    for node_type in dgl_graph.ntypes:
        selected_feature = [
            batched_features[convert_type_and_id_to_name(node_type, node_id)]
            for node_id in dgl_graph.nodes[node_type].data["ID"]
        ]

        time_merged_feature = []
        for node_feat in deepcopy(selected_feature):
            time_merged_feature.append(time_merged_tensor(node_feat, dim=1))
        node_merged_node_feat = reduce(
            lambda a, b: node_merged_tensor(a, b, 0), time_merged_feature
        )
        if dgl_graph.num_nodes(node_type) == 1:  # 当某一个类型的节点数量为1时，要追加一步特殊处理
            node_merged_node_feat = {
                feat_name: feat_tensor
                for feat_name, feat_tensor in time_merged_feature[0].items()
            }
        node_feat_dict[node_type] = node_merged_node_feat
    return node_feat_dict


def time_merged_tensor(merged_list: List, dim: int = 1):
    keys = list(merged_list[0][-1].keys())
    return {
        key: torch.cat(
            [merged_list[index][-1][key] for index in range(len(merged_list))], dim
        )
        for key in keys
    }


def dgl_graph_join_feature(sg: DGLGraph, subfeature: Dict) -> DGLGraph:
    for ntype, feat_dict in subfeature.items():
        for feat_name, feat in feat_dict.items():
            sg.nodes[ntype].data[feat_name] = feat.to(sg.device)
    return sg


def convert_name_to_type_and_id(node_name: str):
    return node_name.split("/")


def convert_type_and_id_to_name(node_type: str, node_id):
    return node_type + "/" + str(int(node_id))


def node_merged_tensor(a, b, dim):
    # TODO python的Reduce性能很差，下个版本应该改掉这个函数
    for key in a:
        # 对于时间步长不足的节点，补零处理
        if a[key].shape[1] > b[key].shape[1]:
            b[key] = F.pad(
                b[key], (0, 0, 0, a[key].shape[1] - b[key].shape[1]), "constant", 0
            )
        elif a[key].shape[1] < b[key].shape[1]:
            a[key] = F.pad(
                a[key], (0, 0, 0, b[key].shape[1] - a[key].shape[1]), "constant", 0
            )
        a[key] = torch.cat([a[key], b[key]], dim=dim)
    return a


def dgl_graph_to_graph_dict(
    dgl_graph: DGLGraph, hetero_feat_dim: Dict
) -> Tuple[Dict, Dict]:
    struc_dict = extract_struc(dgl_graph)
    feat_dict = extract_feat(dgl_graph, hetero_feat_dim)
    return struc_dict, feat_dict


def extract_struc(dgl_graph: DGLGraph) -> Dict:
    struc_dict = {}
    for (src_type, e_type, dst_type) in dgl_graph.canonical_etypes:
        src_id_, dst_id_ = dgl_graph.edges(etype=(src_type, e_type, dst_type))
        src_id = dgl_graph.ndata["ID"][src_type][src_id_].long()
        dst_id = dgl_graph.ndata["ID"][dst_type][dst_id_].long()
        edge_time = (
            dgl_graph.edata["time"][(src_type, e_type, dst_type)]
            if len(dgl_graph.canonical_etypes) > 1
            else dgl_graph.edata["time"]
        )
        struc_dict[(src_type, e_type, dst_type)] = (src_id, dst_id, edge_time)
    return struc_dict


def extract_feat(dgl_graph: DGLGraph, hetero_feat_dim: Dict) -> Dict:
    feat_dict = {}
    dgl_feat = dgl_graph.ndata["state"]
    for ntype in dgl_feat.keys():
        feat_dim_dict = hetero_feat_dim[ntype]
        feats = extract_feat_tool(ntype, dgl_graph, dgl_feat, feat_dim_dict)
        feat_dict[ntype] = feats
    return feat_dict


# TODO 这个函数要单独加入测试
def extract_feat_tool(
    ntype, dgl_graph: DGLGraph, dgl_feat: Dict, feat_dim_dict: Dict
) -> Dict:
    feats: Dict = defaultdict(dict)
    for i, node_id in enumerate(dgl_graph.ndata["ID"][ntype]):
        end_idx = 0
        node_id = int(node_id)
        for feat_name, feat_dim in feat_dim_dict.items():
            feats[node_id][feat_name] = (
                dgl_feat[ntype][i, :, end_idx : end_idx + feat_dim]
                if len(dgl_feat[ntype].shape) == 3
                else dgl_feat[ntype][i, end_idx : end_idx + feat_dim]
            )
            end_idx += feat_dim
    return feats
