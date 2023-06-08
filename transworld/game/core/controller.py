from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Dict, List, Union, Optional, Tuple
from functools import singledispatchmethod

from game.core.node import Node
from game.core.typing import *


class Controller:
    instance = None
    first_init = False

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if self.first_init:
            pass
        else:
            # real initialization
            self.first_init = True
            self.process_table: Dict[str, "Node"] = dict()
            self.dead_process_table: Dict[str, Tuple] = dict()

    def create_single_node(self, process_name: str):
        if process_name not in self.process_table:
            self.process_table[process_name] = Node(process_name)

    def distributed_storage(self, process_name_list: List) -> None:
        for process_name in process_name_list:
            self.create_single_node(process_name)

    def create_node(self, process_name: str):
        self.create_single_node(process_name)

    def retrieve_node(self, process_name: str, timestamp: float) -> "Node":
        if process_name in self.process_table:
            return self.process_table[process_name]
        elif process_name in self.dead_process_table:
            if self.dead_process_table[process_name][-1] > timestamp:
                raise ValueError(f"node is died under currnt retrieve time")
            else:
                return self.dead_process_table[process_name][0]
        else:
            raise KeyError(f"process {process_name} has not beed created!")

    def retrieve_node_structure(
        self, retrieve_message: StructureRetrieveFormat
    ) -> Dict:
        check_structure_retrieve_format(retrieve_message)
        process_ = self.retrieve_node(retrieve_message.name, retrieve_message.timestamp)
        return process_.retrieve_structure(
            timestamp=retrieve_message.timestamp,
            rank_order=retrieve_message.rank_order,
        )

    @singledispatchmethod
    def check_process_exist(self, process_: Union[Tuple, List[Tuple]]) -> bool:
        raise NotImplementedError

    @check_process_exist.register
    def _(self, process_: tuple) -> bool:
        relation_, process_name = process_
        if process_name in self.process_table:
            return True
        else:
            return False

    @check_process_exist.register(list)
    def _(self, process_list: list) -> bool:
        not_created_process_list = []
        for process_ in process_list:
            if not self.check_process_exist(process_):
                not_created_process_list.append(process_)
        if len(not_created_process_list) == 0:
            return True
        else:
            Warning(f"process {not_created_process_list} has not been created!")
            return False

    def set_first_appear_time(self, data: Union[Tuple, List], timestamp: float):
        if isinstance(data, List):
            for tuple_ in data:
                _, connected_process = tuple_
                self.process_table[connected_process].set_node_appear_time(timestamp)
        else:
            _, connected_process = data
            self.process_table[connected_process].set_node_appear_time(timestamp)

    def update_node(self, update_message: StructureCreateFormat):
        check_structure_create_format(update_message)
        process_ = self.retrieve_node(update_message.name, update_message.timestamp)
        if not self.check_process_exist(update_message.data):
            raise NotImplementedError(f"you must created the process first!")

        if update_message.operator == "add":
            self.set_first_appear_time(update_message.data, update_message.timestamp)
            process_.create_address(
                timestamp=update_message.timestamp, address_=update_message.data
            )
        elif update_message.operator == "delete":
            process_.delete_address(
                timestamp=update_message.timestamp, address_=update_message.data
            )
        else:
            raise KeyError(f"{update_message.operator} has not be implemented!")

    def delete_node(self, delete_message: StructureDeleteFormat):
        check_structure_delete_format(delete_message)
        self.dead_process_table[delete_message.name] = (
            self.process_table.pop(delete_message.name),
            delete_message.timestamp,
        )
        for process_name in self.process_table.keys():
            self.process_table[process_name].delete_node(
                timestamp=delete_message.timestamp, node_name=delete_message.name
            )

    def create_feature(self, create_feature_message: FeatureCreateFormat) -> None:
        check_feature_create_format(create_feature_message)
        process_ = self.retrieve_node(
            create_feature_message.name, create_feature_message.timestamp
        )
        process_.create_feature(
            timestamp=create_feature_message.timestamp,
            feature=create_feature_message.data,
        )

    def retrieve_feature(
        self, retrieve_feature_message: FeatureRetrieveFormat
    ) -> Dict[str, List[np.ndarray]]:
        process_ = self.retrieve_node(
            retrieve_feature_message.name, retrieve_feature_message.timestamp
        )
        return {
            retrieve_feature_message.name: process_.retrieve_feature(
                timestamp=retrieve_feature_message.timestamp,
                retrieve_name=retrieve_feature_message.retrieve_name,
                look_back_step=retrieve_feature_message.look_back_step,
            )
        }

    def crud_structure(self, operator: str, operation_list: List):
        return_list = []
        for data in operation_list:
            if operator == "create":
                self.create_node(data)
            elif operator == "retrieve":
                return_list.append(self.retrieve_node_structure(data))
            elif operator == "update":
                self.update_node(data)
            elif operator == "delete":
                self.delete_node(data)
            else:
                raise NotImplementedError

        if operator == "retrieve":
            return return_list

    def crud_feature(self, operator: str, operation_list: List):
        return_list = []
        for data in operation_list:
            if operator == "create":
                self.create_feature(data)
            elif operator == "retrieve":
                return_list.append(self.retrieve_feature(data))
            else:
                raise NotImplementedError
        return return_list

    @singledispatchmethod
    def retrieve_alive_node(self, timestamp: Union[float, List[float]]) -> List:
        raise NotImplementedError

    @retrieve_alive_node.register
    def _(self, timestamp: float) -> List:
        assert type(timestamp) == float, "timestamp must by float type"
        node_list = []
        for alive_node_name, node in self.process_table.items():
            if node.start_appear_time <= timestamp:
                node_list.append(alive_node_name)

        for death_node_name, tuple_ in self.dead_process_table.items():
            # when appear time is latter than retrieve time and death time is before than retrieve time
            if tuple_[0].start_appear_time <= timestamp and timestamp <= tuple_[1]:
                node_list.append(death_node_name)

        return node_list

    @retrieve_alive_node.register
    def _(self, timestamp_list: list) -> List:
        return [self.retrieve_alive_node(timestamp) for timestamp in timestamp_list]

    def run(self):
        return super().run()

    @classmethod
    def reset(cls):
        cls.instance = None
        cls.first_init = False
