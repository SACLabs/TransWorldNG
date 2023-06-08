from typing import Dict, Set, List, Tuple, Any, Union, Callable, Optional
import copy
import numpy as np
import bisect


from functools import singledispatchmethod
from collections import defaultdict
from queue import Queue


from game.core.config import TIMELENGTH


class Node:
    instantces: Dict[str, "Node"] = dict()
    first_init: Set[str] = set()

    def __new__(cls, name: str, *args, **kwargs):
        if name not in cls.instantces:
            cls.instantces[name] = super().__new__(cls)
        return cls.instantces[name]

    def __init__(self, name: str):
        if name not in self.first_init:
            self.first_init.add(name)
            ###### real initialization #####
            self.name = name
            self.address_timestamp_queue: Queue[float] = Queue(maxsize=TIMELENGTH + 1)
            self.feature_timestamp_queue: Queue[float] = Queue(maxsize=TIMELENGTH + 1)
            self.feature_with_timestamp: Dict[float, Dict[str, np.ndarray]] = dict()
            self.address_book_with_timestamp: Dict[float, Dict] = dict()
            self.start_appear_time = -1
            self.address_book: Dict[str, List] = defaultdict(list)
            self.last_structure_modified_timestamp = -1.0
            self.last_feature_modified_timestamp = -1.0

    def chenck_and_copy_address(self, operation_timestamp):
        if operation_timestamp != self.last_structure_modified_timestamp:

            if self.address_timestamp_queue.full():
                destroy_key = self.address_timestamp_queue.get()
                self.address_book_with_timestamp.pop(destroy_key)

            self.address_timestamp_queue.put(operation_timestamp)
            self.address_book_with_timestamp[operation_timestamp] = copy.deepcopy(
                self.address_book
            )
            self.last_structure_modified_timestamp = operation_timestamp

        else:
            self.address_book_with_timestamp[operation_timestamp] = copy.deepcopy(
                self.address_book
            )

    def set_node_appear_time(self, timestamp):
        if self.start_appear_time < 0 or self.start_appear_time >= timestamp:
            self.start_appear_time = timestamp

    def create_address(self, timestamp, address_: Union[List, Tuple]):
        assert (
            timestamp >= self.last_structure_modified_timestamp
        ), "timestamp must bigger than last structure modifed time"
        if self.start_appear_time < 0:
            self.start_appear_time = timestamp
        Node.add_remove_operation(
            address_, timestamp, self.address_book, Node.add_address
        )
        self.chenck_and_copy_address(timestamp)

    def retrieve_structure(
        self, timestamp: float, rank_order: int = 1, node_name: Optional[str] = None
    ) -> Dict:
        # 获取end_time位置的graph结构, rank_order表示的是几阶子图
        structure_dict: Dict[str, Dict] = dict()
        search_node_name = self.name if node_name is None else node_name
        structure_history_timestamp_list = list(
            Node.instantces[search_node_name].address_book_with_timestamp.keys()
        )
        if len(structure_history_timestamp_list) == 0:
            return {search_node_name: {}}

        sorted_timestamp_list, index = Node.search_nearest_key(
            structure_history_timestamp_list, timestamp
        )

        if index == 0 and timestamp < sorted_timestamp_list[0]:
            Warning("Current Node is an asolate node!")
            structure_dict.update({search_node_name: {}})
            return structure_dict

        nearest_timestamp = structure_history_timestamp_list[max(0, index - 1)]
        connected_address_book = Node.instantces[
            search_node_name
        ].address_book_with_timestamp[nearest_timestamp]
        if rank_order == 1:
            structure_dict.update({search_node_name: connected_address_book})
            return structure_dict
        else:
            for connected_node_name in connected_address_book.keys():
                structure_dict.update(
                    self.retrieve_structure(
                        timestamp, rank_order - 1, connected_node_name
                    )
                )
            structure_dict.update({search_node_name: connected_address_book})
            return structure_dict

    def delete_address(self, timestamp: float, address_: Union[List, Tuple]) -> None:
        # remove算子没有这么简单，需要在一个列表中移除掉一个address
        Node.add_remove_operation(
            address_,
            timestamp,
            self.address_book,
            Node.remove_address,
        )
        self.chenck_and_copy_address(timestamp)

    def delete_node(self, timestamp: float, node_name: str) -> None:
        # 这个是移除当前码本中，所有与这个node相链接的边
        if node_name in self.address_book:
            self.address_book.pop(node_name)
            self.chenck_and_copy_address(timestamp)

    def bak_feature(self, timestamp: float) -> None:
        if self.feature_timestamp_queue.full():
            destroy_key = self.feature_timestamp_queue.get()
            self.feature_with_timestamp.pop(destroy_key)

        self.feature_timestamp_queue.put(timestamp)
        self.feature_with_timestamp[timestamp] = copy.deepcopy(self.feature)
        self.last_structure_modified_timestamp = timestamp

    def create_feature(self, timestamp: float, feature: Dict[str, np.ndarray]) -> None:
        # 先更新，后设置副本，这样就不会把None放进去了
        assert (
            timestamp >= self.last_feature_modified_timestamp
        ), "timestamp must bigger than last feature modified timestamp!"
        self.feature = feature
        self.bak_feature(timestamp)

    def retrieve_feature(
        self,
        timestamp: float,
        retrieve_name: Optional[List[str]],
        look_back_step: Optional[int] = None,
    ) -> List:
        feature_history_timestamp_list = list(self.feature_with_timestamp.keys())
        if len(feature_history_timestamp_list) == 0:
            return []
        else:
            sorted_list, start_index = Node.search_nearest_key(
                feature_history_timestamp_list, timestamp
            )
            if start_index == 0:
                if sorted_list[0] == timestamp:
                    return [(timestamp, self.feature_with_timestamp[timestamp])]
                else:
                    return []
            else:
                if look_back_step is not None:
                    legal_timestamp_list = sorted_list[
                        max(0, start_index - look_back_step) : start_index
                    ]
                else:
                    legal_timestamp_list = sorted_list[0:start_index]

                if retrieve_name is None:
                    return [
                        (_timestamp, self.feature_with_timestamp[_timestamp])
                        for _timestamp in legal_timestamp_list
                    ]
                else:
                    return [
                        (
                            _timestamp,
                            {
                                feat_name: self.feature_with_timestamp[_timestamp][
                                    feat_name
                                ]
                                for feat_name in retrieve_name
                            },
                        )
                        for _timestamp in legal_timestamp_list
                    ]

    @staticmethod
    def search_nearest_key(src_list, target_value):
        # 对src_list 进行排序，然后采用二分查找，O(nlogn + logn)
        sorted_list = sorted(src_list)
        find_index = bisect.bisect_left(src_list, target_value)
        return sorted_list, find_index

    @staticmethod
    def remove_address(src_list, remove_tuple):
        relation, _ = remove_tuple
        rm_relations = [tuple_ for tuple_ in src_list if relation in tuple_]
        if len(rm_relations) > 1:
            raise ValueError("Removed adddress should less then one.")
        elif len(rm_relations) == 0:
            Warning(f"{remove_tuple} not exist")
        else:
            src_list.remove(rm_relations[0])

    @staticmethod
    def add_address(src_list, add_tuple):
        if add_tuple not in src_list:
            list.append(src_list, add_tuple)

    @singledispatchmethod
    @staticmethod
    def add_remove_operation(
        address_: Union[List, Tuple],
        timestamp,
        source_object: Dict,
        Operator: Callable,
    ) -> None:
        raise NotImplementedError

    @add_remove_operation.register
    @staticmethod
    def _(address_: list, timestamp, source_object: Dict, Operator: Callable):
        for tuple_ in address_:
            Node.add_remove_operation(tuple_, timestamp, source_object, Operator)

    @add_remove_operation.register
    @staticmethod
    def _(address_: tuple, timestamp, source_object: Dict, Operator: Callable):
        relation, connected_process = address_
        Operator(source_object[connected_process], (relation, timestamp))
        if len(source_object[connected_process]) == 0:
            source_object.pop(connected_process)

    @classmethod
    def reset(cls):
        cls.first_init = set()
        cls.instantces = dict()
