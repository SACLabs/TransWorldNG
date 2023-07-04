from copy import deepcopy
import torch
import torch.nn as nn
from dgl import DGLGraph
import dgl.nn.pytorch as dglnn
from .linear import HomoToHeteroLinear
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Union, Optional, Tuple, Callable, Generic
from torch import Tensor
from abc import ABC, abstractmethod
from game.operator.transform import dgl_graph_to_graph_dict


class OneShotGenerator(nn.Module):
    """Directly generate new graph from a graph
    注意这种模型无法使用图补全来预测，而必须使用图生成的方法，因为图补全无法产生delete操作

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, List[str]], DGLGraph]:
        new_graph = self.generate_graph(node_name_at_time, subgraph, node_repr)
        node_states = self.generate_states(node_name_at_time, subgraph, node_repr)
        actions = self.graph_to_actions(new_graph)
        return node_states, actions, new_graph

    @abstractmethod
    def generate_graph(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
    ) -> DGLGraph:
        # NN based graph generator
        pass

    @abstractmethod
    def generate_states(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # NN based graph generator
        pass

    def graph_to_actions(self, subgraph: DGLGraph):
        # TODO
        # new  graph to actions
        NotImplementedError


class SequentialGenerator(nn.Module):
    """Generate actions step by step from a graph

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, List[str]], DGLGraph]:
        actions, node_states = self.generate_actions_and_state(
            node_name_at_time, subgraph, node_repr
        )
        delta_graph = self.actions_to_graph(actions)
        return node_states, actions, delta_graph

    @abstractmethod
    def generate_actions_and_state(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Tensor]]:
        # TODO NN based actions generator
        pass

    def actions_to_graph(self, actions: Dict[str, List[str]]):
        # TODO
        # actions to graph
        NotImplementedError


class RuleBasedGenerator(nn.Module):
    def __init__(
        self,
        hetero_feat_dim: Dict,
        repr_dim: int,
        pred_dim: Dict[str, int],
        scalers: Optional[Dict] = None,
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hetero_feat_dim = hetero_feat_dim
        self.state_projector = HomoToHeteroLinear(
            repr_dim,
            pred_dim,
            activation=activation,
            output_activation=output_activation,
        )
        self.scalers = scalers

    def register_rule(self, rule_func) -> None:
        self.rule_func = rule_func

    def generate_actions_and_state(
        self,
        operate_function,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, Tensor], Dict[str, List[str]]]:
        node_states, node_state_scaled = self.generate_state(node_repr)
        actions = operate_function(
            self.rule_func,
            node_name_at_time,
            subgraph,
            node_state_scaled,
            *args,
            **kwargs,
        )
        return node_states, actions

    def generate_state(
        self, node_repr: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        node_state = self.state_projector(node_repr)
        if self.scalers is not None:
            node_state_scaled = {
                ntype: sklearn_scaling(self.scalers[ntype], nstate)
                for ntype, nstate in node_state.items()
            }
        else:
            node_state_scaled = node_state
        return node_state, node_state_scaled

    def generate_actions(
        self,
        hook_fn,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_states: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, List[str]]:
        subgraph.ndata["state"] = node_states
        struc_dict, feat_dict = dgl_graph_to_graph_dict(subgraph, self.hetero_feat_dim)
        actions = hook_fn(node_name_at_time, struc_dict, feat_dict, *args, **kwargs)
        return actions

    def forward(
        self,
        node_name_at_time: List[str],
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, List[str]], DGLGraph]:
        outgraph = deepcopy(subgraph)
        node_states, actions = self.generate_actions_and_state(
            self.generate_actions,
            node_name_at_time,
            outgraph,
            node_repr,
            *args,
            **kwargs,
        )
        outgraph.ndata["state"] = node_states
        return actions, outgraph

    def output_graph_dict(
        self,
        subgraph: DGLGraph,
        node_repr: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        node_states = self.generate_state(node_repr)
        struc_dict, feat_dict = dgl_graph_to_graph_dict(subgraph, self.hetero_feat_dim)
        return struc_dict, feat_dict


def sklearn_scaling(scaler, X: Tensor) -> Tensor:
    X_need_scale = X[:, :-1]
    X_no_scale = X[:, [-1]]
    X_scaled = (
        X_need_scale - torch.tensor(scaler.min_).float().to(X_need_scale.device)
    ) / torch.tensor(scaler.scale_).float().to(X_need_scale.device)
    X = torch.cat([X_scaled, X_no_scale], dim=1)
    return X
