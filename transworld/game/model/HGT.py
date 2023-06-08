import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph
import dgl.nn.pytorch as dglnn
from .linear import HeteroToHomoLinear
from typing import Any, Dict, List, Union, Optional, Tuple
from torch import Tensor


class HGT(nn.Module):
    def __init__(
        self,
        in_dim: Dict[str, int],
        n_ntypes: int,
        n_etypes: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.2,
        use_norm=True,
        activation: Optional[nn.Module] = None,
    ):
        """A simplest heterogenous graph neural network model.

        Args:
            in_dim (Dict[str, int]): The dimensions of different node features
            out_dim (Dict[str, int]): The dimensions of different node represemtation learned by HGT
            n_ntypes (int): Num of node types
            n_etypes (int): Num of edge types
            hid_dim (int): Dimension of universal feature sapce
            n_layers (int): Num of GNN layers
            n_heads (int): Num of attention heads
            dropout (float, optional): dropout. Defaults to 0.2.
            use_norm (bool, optional): normalization. Defaults to True.
        """
        super(HGT, self).__init__()
        self.gnns = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hid_dim = hid_dim  # 定义统一特征空间维数，将不同类型的特征映射到统一特征空间
        self.n_layers = n_layers
        self.hetero_input_projector = HeteroToHomoLinear(
            in_dim, hid_dim, activation=activation
        )
        in_size_list = [hid_dim] + [hid_dim * n_heads for _ in range(n_layers - 1)]
        for in_size in in_size_list:
            self.gnns.append(
                dglnn.HGTConv(
                    in_size=in_size,
                    head_size=hid_dim,
                    num_heads=n_heads,
                    num_ntypes=n_ntypes,
                    num_etypes=n_etypes,
                    dropout=dropout,
                    use_norm=use_norm,
                )
            )
            self.linears.append(
                nn.Linear(hid_dim * n_heads, hid_dim * n_heads, bias=True)
            )
            self.bns.append(nn.BatchNorm1d(hid_dim * n_heads))
        self.activation = activation

    def forward(
        self, subgraph: DGLGraph, *, hetero_output: bool = True
    ) -> Dict[str, Tensor]:
        """_summary_

        Args:
            subgraph (DGLGraph): a suggraph that can be seen by a agent
            feat_name (str): user-defined feature name
            hetero_output (bool, optional): user can use heterogenous output or homogenous one. Defaults to True. Note that this parameter is not ready in this version.

        Returns:
            _type_: _description_
        """
        with subgraph.local_scope():
            # 先将不同类型的节点特征映射到同一空间
            assert len(subgraph.ntypes) > 1, ValueError(
                "HGT only support heterogenous graph, but the input graph is homogenoous one."
            )
            subgraph.ndata["repr"] = self.hetero_input_projector(
                subgraph.ndata["state"]
            )
            # 然后转成同质图
            g = dgl.to_homogeneous(subgraph, ndata=["repr"])
            ntype_indicator = g.ndata[dgl.NTYPE]
            etype_indicator = g.edata[dgl.ETYPE]
            h = g.ndata["repr"].squeeze()
            for i in range(self.n_layers):
                h = self.gnns[i](g, h, ntype_indicator, etype_indicator)
                if self.activation is not None:
                    h = self.activation(h)
                h = self.bns[i](h)
                h = self.linears[i](h)
            if hetero_output is True:
                h = {
                    ntype: h[ntype_indicator == i]
                    for i, ntype in enumerate(subgraph.ntypes)
                }
            return h
