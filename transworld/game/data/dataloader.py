import torch
import math
from collections import defaultdict, ChainMap
from copy import deepcopy
from torch import Tensor
from typing import (
    Dict,
    List,
    Tuple,
)
from game.operator.transform import game_to_dgl
from game.core.typing import (
    FeatureRetrieveFormat,
    StructureRetrieveFormat,
)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        operation_fn,
        full_graph: bool = True,
        need_negative_sampling: bool = False,
        **kwargs
    ) -> None:
        self.operation_fn = operation_fn
        self.dataset = dataset
        self.device = dataset.device
        self.full_graph = full_graph
        self.train_mode = dataset.train_mode
        self.need_negative_sampling = need_negative_sampling
        super().__init__(self.dataset, collate_fn=self.collate, **kwargs)

    def collate(self, batched_timestamp: List):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(
                math.ceil(len(batched_timestamp) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(batched_timestamp))
            batched_timestamp = batched_timestamp[iter_start:iter_end]
        if self.train_mode is True:
            batched_cur_graph, batched_next_graph = [], []
            for pair_timestamp in batched_timestamp:
                cur_graph, next_graph = self.collate_tool(list(pair_timestamp))
                batched_cur_graph.append(cur_graph)
                batched_next_graph.append(next_graph)
            return ChainMap(*batched_cur_graph), ChainMap(*batched_next_graph)
        else:
            batched_cur_graph = self.collate_tool(batched_timestamp)
            return ChainMap(*batched_cur_graph)

    def collate_tool(self, batched_timestamp: List[float], max_step: int = 1):
        # TODO 后期加入测试
        nodes_pool_at_time = self.operation_fn(
            operation_type="alive_node",
            operator="retrieve",
            operation_list=batched_timestamp,
        )
        if self.train_mode is True:
            union_nodes = set.intersection(*map(set, nodes_pool_at_time))
            nodes_pool_at_time = [union_nodes for _ in range(len(batched_timestamp))]
        batched_structures_ = [
            self.operation_fn(
                operation_type="structure",
                operator="retrieve",
                operation_list=[
                    StructureRetrieveFormat(name=node, timestamp=ts, rank_order=1)
                    for node in nodes_pool_at_time[i]
                ],
            )
            for i, ts in enumerate(batched_timestamp)
        ]
        batched_structures = [delete_empty_structure(bs) for bs in batched_structures_]

        batched_features_ = [
            self.operation_fn(
                operation_type="feature",
                operator="retrieve",
                operation_list=[
                    FeatureRetrieveFormat(
                        name=node, timestamp=ts, look_back_step=max_step
                    )
                    for node in nodes_pool_at_time[i]
                ],
            )
            for i, ts in enumerate(batched_timestamp)
        ]
        batched_features = [dictify_feature_list(bf) for bf in batched_features_]

        batched_structures, batched_features = self.preprocess(
            batched_structures, batched_features
        )
        # if self.need_negative_sampling is True:
        #     TODO define negative_sampling func

        batched_dgl_graph = [
            game_to_dgl(
                batched_structure,
                batched_feature,
                str(batched_timestamp[i]),
                self.full_graph,
                self.device,
            )
            for i, (batched_structure, batched_feature) in enumerate(
                zip(batched_structures, batched_features)
            )
        ]
        return batched_dgl_graph

    # def worker_init_fn_(self, worker_id):
    #     TODO: Xuhong 实现多进程加载数据，正在考虑是使用pytroch自带的多进程还是MPI实现
    #     worker_info = torch.utils.data.get_worker_info()
    #     dataset = worker_info.dataset  # the dataset copy in this worker process
    #     # configure the dataset to only process the split workload
    #     per_worker = int(len(dataset.dataset) // float(worker_info.num_workers))
    #     worker_id = worker_info.id
    #     dataset.dataset = dataset.dataset[worker_id*per_worker:(worker_id+1)*per_worker]
    #     print('dataset',dataset.dataset)

    def preprocess(
        self, batched_structure: List, batched_feature: List
    ) -> Tuple[List, List]:
        # 这个函数可以让用户放入一些前处理函数，比如check前处理规则等等
        return batched_structure, batched_feature


def dictify_feature_list(batched_features_list: List) -> Dict[str, List]:
    # TODO 后期加入测试
    # 将feature不同时刻查询的列表合并为字典
    batched_features: Dict = defaultdict(list)
    for features in deepcopy(batched_features_list):
        for node_name, feats in features.items():
            for feat in feats:
                if (batched_features[node_name] == []) or (
                    feat[0] not in list(zip(*batched_features[node_name]))[0]
                ):
                    feat = (
                        feat[0],
                        {key: tensor_unsqueeze(val) for key, val in feat[1].items()},
                    )
                    batched_features[node_name].append(feat)
    return batched_features


def tensor_unsqueeze(tensor: Tensor, target_shape: int = 3):
    while len(tensor.shape) < target_shape:
        tensor.unsqueeze_(0)
    assert len(tensor.shape) == 3, ValueError(
        "Feature shape must equals to 3. But got {}".format(tensor.shape)
    )
    return tensor


def delete_empty_structure(batched_structures: List[Dict]) -> List[Dict]:
    batched_structures = [
        structure
        for structure in batched_structures
        if list(structure.values())[0] != {}
    ]
    return batched_structures
