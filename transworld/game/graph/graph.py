from collections import defaultdict, namedtuple, OrderedDict
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import re
import dgl
import torch
from dgl import DGLGraph
from torch import Tensor

from game.core.controller import Controller
from game.core.typing import (
    FeatureCreateFormat,
    StructureCreateFormat,
    StructureDeleteFormat,
)
from game.operator.transform import (
    dgl_graph_to_graph_dict,
    extract_feat,
    convert_type_and_id_to_name,
)

from game.core.node import Node
SubGraph = namedtuple("SubGraph", ["seed_node_id", "seed_node_type", "dgl_graph"])


class Graph(object):
    """_summary_

    Args:
        object (_type_): game.Graph Object
    """

    def __init__(
        self,
        edge_dict: Dict,
        feat_dict: Dict,
        build_directed: bool = True,
    ):

        """Internal constructor for creating a game.Graph.

        Args:
            edge_dict (Dict): Edge dictionary object.
            feat_dict (Dict): Node Feature dictionary object.
            build_directed (bool, optional): If Ture, build a directed graph; if False, build a undirected graph. Defaults to True.

        Examples:
            TODO：后续将输入数据修改为dataclass的形式
            边字典：第一个tensor为源节点列表u，第二个为目标节点v，第三个为时间戳t，第四个tensor(可选)为边上的特征Fe
            edge_dict = {
                ('Car', 'follows', 'Car'): (torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2321.0, 2323.1]), torch.tensor([0.06, 1.1])),
                ('Car', 'locates', 'Lane'): (torch.tensor([1, 1]), torch.tensor([1, 2]), torch.tensor([231.0, 233.2]), torch.tensor([1.77, 2.1])),
                ('Lane', 'connects', 'Lane'): (torch.tensor([0, 3]), torch.tensor([3, 4]), torch.tensor([1321.0, 2333.5]), torch.tensor([1.88, -0.16]))
            }
            节点特征字典：
            feat_dict = {
                        'Car':{
                            0:{'time':[torch.tensor([0.0]), torch.tensor([1.1]), torch.tensor([2.7]), torch.tensor([5.9])],
                                'feature':[torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3])]
                            },
                            1:{'time':[torch.tensor([0.0]), torch.tensor([1.1]), torch.tensor([2.7]), torch.tensor([5.9])],
                                'feature':[torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3])]
                            },
                        },
                        'Lane':{
                            0:{'time':[torch.tensor([0.0]), torch.tensor([1.1]), torch.tensor([2.7]), torch.tensor([5.9])],
                                'feature':[torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3])]
                            },
                            1:{'time':[torch.tensor([0.0]), torch.tensor([1.1]), torch.tensor([2.7]), torch.tensor([5.9])],
                                'feature':[torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3])]
                            },
                        }
                    }
            graph = Graph(graph_dict, feat_dict, build_directed=True)
        """
        if (edge_dict is not None) and (feat_dict is not None):
            self.is_directed = build_directed
            self.is_attr_graph = False if feat_dict is None else True
            self.dgl_graph = edge_dict_to_dgl(edge_dict)
            self.controller = Controller()
            self.node_list = self.store_data(self.dgl_graph, feat_dict)
            self.timestamps = self.collect_timestamps(self.dgl_graph)
            self.hetero_feat_dim = heterogenous_feature_parsing(feat_dict)

    @property
    def is_attr(self) -> bool:
        return self.is_attr_graph

    @property
    def num_ntypes(self) -> int:
        return len(self.dgl_graph.ntypes)

    @property
    def num_etypes(self) -> int:
        return len(self.dgl_graph.etypes)

    @property
    def num_timestamps(self) -> int:
        return len(self.timestamps)

    def operate(self, operation_type: str, operator: str, operation_list: List) -> List:
        results = []
        if operation_type == "feature":
            results = self.controller.crud_feature(operator, operation_list)
        elif operation_type == "structure":
            results = self.controller.crud_structure(operator, operation_list)
        elif operation_type == "alive_node":
            results = self.controller.retrieve_alive_node(operation_list)
        else:
            ValueError("Please use correct operation_type name")
        return results

    def store_data(self, dgl_graph_: DGLGraph, feat_dict: Dict):
        # call controller: create isolated nodes
        node_list = []
        for ntype in dgl_graph_.ntypes:
            node_list.extend(
                [f"{ntype}/{node_id}" for node_id in dgl_graph_.nodes(ntype).tolist()]
            )

        self.controller.crud_structure(operator="create", operation_list=node_list)

        # call controller: add subgraph to each node
        _update_edge_events = partial(self._update_edge_events, dgl_graph_)

        edge_events = list(
            map(
                _update_edge_events,
                [
                    (int(node_id), ntype)
                    for ntype in dgl_graph_.ntypes
                    for node_id in dgl_graph_.nodes(ntype)
                ],
            )
        )
        self.controller.crud_structure(
            operator="update", operation_list=list(chain.from_iterable(edge_events))
        )

        # call controller: add feature to each node
        feature_events = list(
            map(
                _create_feature_events,
                [
                    (int(node_id), node_type, node_feats)
                    for node_type in feat_dict.keys()
                    for node_id, node_feats in feat_dict[node_type].items()
                ],
            )
        )
        self.controller.crud_feature(
            operator="create", operation_list=list(chain.from_iterable(feature_events))
        )

        return node_list

    def collect_timestamps(self, dgl_graph):
        unique_timestamps = torch.unique(
            torch.cat(list(dgl_graph.edata["time"].values()))
        )
        unique_timestamps, _ = torch.sort(unique_timestamps)
        timestamps = unique_timestamps[1:]  # 去除最早的第一个timestamp，因为没有意义
        return timestamps

    def _update_edge_events(self, dgl_graph_, args: Tuple):
        node_id, ntype = args
        dgl_SG = extract_subgraph(dgl_graph_, node_id, ntype)
        edge_events = dgl_to_structure_event(dgl_SG)
        edge_events = sorted(edge_events, key=lambda x: x.timestamp)
        return edge_events

    def actions_to_game_operations(self, actions: Dict[str, List[str]]) -> None:
        for node_name_n_time, action_list in actions.items():
            for action in action_list:
                if re.search(".*edge", action) is not None:
                    structure_operation = self.action_to_structure_event(
                        node_name_n_time, action
                    )
                    self.controller.crud_structure(
                        operator="update", operation_list=[structure_operation]
                    )
                elif re.search(".*node", action) is not None:
                    node_operation = self.action_to_node_event(node_name_n_time, action)
                    self.controller.crud_structure(
                        operator="delete", operation_list=[node_operation]
                    )
                else:
                    NotImplementedError

    def action_to_structure_event(self, node_name_n_time: str, action: str):
        edge_str = re.findall(r"[(](.*?)[)]", action)[0]
        src_name, etype, dst_name = edge_str.split(",")
        operation = StructureCreateFormat(
            name=src_name,
            timestamp=round(float(node_name_n_time.split("@")[-1]), 6),
            operator=action.split("_")[0],
            data=(etype, dst_name),
        )
        return operation

    def action_to_node_event(self, node_name_n_time: str, action: str):
        node_name = re.findall(r"[(](.*?)[)]", action)[0]
        operation = StructureDeleteFormat(
            name=node_name,
            timestamp=round(float(node_name_n_time.split("@")[-1]), 6),
        )
        return operation

    def states_to_feature_event(
        self,
        time: float,
        changable_feature_names: List[str],
        cur_graph: DGLGraph,
        pred_graph: DGLGraph,
    ):
        cur_feat = extract_feat(cur_graph.cpu(), self.hetero_feat_dim)
        pred_feat = extract_feat(pred_graph.cpu(), self.hetero_feat_dim)

        for ntype in pred_feat.keys():
            for node_id in pred_feat[ntype].keys():
                feat_data = {
                    feat_name: pred_feat[ntype][node_id][feat_name].squeeze()
                    if feat_name in changable_feature_names
                    else cur_feat[ntype][node_id][feat_name].squeeze()
                    for feat_name in pred_feat[ntype][node_id].keys()
                }
                feat_event = FeatureCreateFormat(
                    name=convert_type_and_id_to_name(ntype, node_id),
                    timestamp=time,
                    data=feat_data,
                )
                self.operate("feature", "create", [feat_event])


    @staticmethod
    def reset():
        Controller.reset()
        Node.reset()

def heterogenous_feature_parsing(feat_dict):
    feat_parsing_dict: Dict[str, Dict] = defaultdict(dict)
    for ntype, feat in feat_dict.items():
        for name, tensor in list(feat.values())[0].items():
            dim = tensor.shape[-1] if len(tensor.shape) != 1 else 1
            feat_parsing_dict[ntype][name] = dim
    return feat_parsing_dict


def _create_feature_events(node_item: Tuple):
    feature_events = []
    node_id, node_type, node_feats = node_item
    feat_name_list = list(node_feats.keys())
    assert "time" in feat_name_list, "Feature Dictionary shall include time"
    last_timestamp = 0.  #round(float(node_feats['time'][0].item() - 1.), 6)
    for feats in zip(*node_feats.values()):
        cur_timestamp = round(float(feats[feat_name_list.index("time")].item()), 6)
        data = {
                feat_name: feats[i]
                if feat_name != "time"
                else feats[i].float() - last_timestamp
                for i, feat_name in enumerate(feat_name_list)
                }
        feature_events.append(
            FeatureCreateFormat(
                name=f"{node_type}/{node_id}",
                timestamp=cur_timestamp,
                data=data,
            )
        )
        last_timestamp = cur_timestamp
    feature_events = sorted(feature_events, key=lambda x: x.timestamp)
    return feature_events

 
def dgl_to_structure_event(sub_graph: SubGraph) -> List[StructureCreateFormat]:
    # 这个方法只支持转换子图
    def _id_tensors_to_edge_events(edge_item: Tuple):
        src_id_, dst_id_, e_id = edge_item
        src_id = dgl_graph_.ndata["ID"][src_type][src_id_]
        dst_id = dgl_graph_.ndata["ID"][dst_type][dst_id_]
        event = None
        if src_id == seed_node_id:
            t = dgl_graph_.edata["time"][(src_type, etype, dst_type)][e_id].item()
            event = StructureCreateFormat(
                name=f"{src_type}/{src_id}",
                timestamp=round(float(t), 6),
                operator="add",
                data=(etype, f"{dst_type}/{dst_id}"),
            )
        return event

    edge_events = []
    dgl_graph_ = sub_graph.dgl_graph
    seed_node_type = sub_graph.seed_node_type
    seed_node_id = sub_graph.seed_node_id
    for (src_type, etype, dst_type) in dgl_graph_.canonical_etypes:
        if src_type == seed_node_type:
            src_id_tensor, dst_id_tensor, e_id_tensor = dgl_graph_.edges(
                form="all", etype=(src_type, etype, dst_type)
            )
            edge_events.extend(
                list(
                    map(
                        _id_tensors_to_edge_events,
                        [
                            (src_id.item(), dst_id.item(), e_id)
                            for (src_id, dst_id, e_id) in zip(
                                src_id_tensor, dst_id_tensor, e_id_tensor
                            )
                        ],
                    )
                )
            )
    edge_events = list(filter(None, edge_events))
    return edge_events


def extract_subgraph(dgl_graph_: DGLGraph, node_id: int, node_type: str):

    hop_nodes = defaultdict(list)
    hop_nodes[node_type].extend([node_id])
    for canonical_etype in dgl_graph_.canonical_etypes:
        src_type, _, dst_type = canonical_etype
        if src_type == node_type:
            hop_nodes[dst_type].extend(
                dgl_graph_.successors(node_id, etype=canonical_etype).tolist()
            )
    for k, v in hop_nodes.items():
        hop_nodes[k] = list(set(v))
    dgl_subgraph = dgl.node_subgraph(dgl_graph_, hop_nodes)
    dgl_subgraph.ndata["ID"] = dgl_subgraph.ndata[dgl.NID]

    sub_graph = SubGraph._make([node_id, node_type, dgl_subgraph])
    return sub_graph


def edge_dict_to_dgl(edge_dict: Dict):
    # 仅在用户输入数据的时候使用一次
    # 由于DGLGraph不认识时间tensor和特征tensor，先将时间、拓扑逻辑、边上的特征分离,在组合到DGLGraph中
    dgl_dict = {}
    time_dict = {}
    edge_feat_dict = {}
    for metapath, uvtf in edge_dict.items():
        uv = (uvtf[0], uvtf[1])
        t = uvtf[2]
        if len(uvtf) > 3:
            f = uvtf[3]
            edge_feat_dict[metapath] = f
        dgl_dict[metapath] = uv
        time_dict[metapath] = t
    dgl_graph_ = dgl.heterograph(dgl_dict)
    dgl_graph_.edata["time"] = time_dict
    if len(edge_feat_dict) > 0:
        dgl_graph_.edata["feat"] = edge_feat_dict
    return dgl_graph_

