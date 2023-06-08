import torch
import torch.nn as nn
from dgl import DGLGraph
from typing import List
import torch.nn.functional as F
from torch import Tensor


class GraphStateLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GraphStateLoss, self).__init__()

    def forward(self, pred_graph: DGLGraph, tar_graph: DGLGraph):
        loss = torch.Tensor([0.0]).to(pred_graph.device)
        total_node_type = get_intersection(pred_graph.ntypes, tar_graph.ntypes)
        for node_type in total_node_type:
            if tensor_is_equal(
                pred_graph.ndata["ID"][node_type], tar_graph.ndata["ID"][node_type]
            ):
                pred_state = pred_graph.ndata["state"][node_type]
                tar_state = (
                    tar_graph.ndata["state"][node_type][:, -1, :]
                    if len(tar_graph.ndata["state"][node_type].shape) > 2
                    else tar_graph.ndata["state"][node_type]
                )
            else:
                node_id_of_pred = pred_graph.ndata["ID"][node_type].tolist()
                node_id_of_tar = tar_graph.ndata["ID"][node_type].tolist()
                comm_node_id = get_intersection(node_id_of_pred, node_id_of_tar)
                pred_state = torch.cat(
                    [
                        pred_graph.ndata["state"][node_type][idx].unsqueeze(0)
                        for idx, nid in enumerate(node_id_of_pred)
                        if nid in comm_node_id
                    ]
                )
                tar_state = torch.cat(
                    [
                        tar_graph.ndata["state"][node_type][idx, -1, :].unsqueeze(0)
                        for idx, nid in enumerate(node_id_of_tar)
                        if nid in comm_node_id
                    ]
                )
            loss_ = F.mse_loss(pred_state, tar_state)  # 只计算最后一个step的state，历史数据不需要计算
            loss = loss + loss_
        return loss


def get_intersection(ListA: List, ListB: List) -> List:
    return list(set(ListA).intersection(set(ListB)))


def tensor_is_equal(TensorA: Tensor, TensorB: Tensor) -> bool:
    if len(TensorA) == len(TensorB) and (TensorA == TensorB).all().item():
        return True
    else:
        return False
