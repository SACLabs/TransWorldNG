from typing import List, Tuple, Union, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureCreateFormat:
    name: str
    timestamp: float
    data: Dict[str, np.ndarray]


def check_feature_create_format(message: FeatureCreateFormat):
    assert (
        type(message.name) == str
    ), "the type of FeatureCreateFormat.name must be string"
    assert (
        type(message.timestamp) == float
    ), "the type of FeatureCreateFormat.timestamp must be float"
    # for key, value in message.data.items():
    #     assert (
    #         type(key) == str
    #     ), "the type of FeatureCreateFormat.data.keys() must be string"
    #     assert (
    #         type(value) == np.ndarray
    #     ), "the type of FeatureCreateFormat.data.values() must be np.ndarray"


@dataclass
class FeatureRetrieveFormat:
    name: str
    timestamp: float
    retrieve_name: Optional[List[str]] = None
    look_back_step: Optional[int] = None


def check_feature_retrieve_format(message: FeatureRetrieveFormat):
    assert (
        type(message.name) == str
    ), "the type of FeatureRetrieveFormat.name must be string"
    assert (
        type(message.timestamp) == float
    ), "the type of FeatureRetrieveFormat.timestamp must be float"
    assert (
        type(message.retrieve_name) == list or message.retrieve_name is None
    ), "the type of FeatureRetrieveFormat.retrieve_name must be Optional[List[str]]"
    assert (
        type(message.look_back_step) == int or message.look_back_step is None
    ), "the type of FeatureRetrieveFormat.look_back_step must be Optional[int]"


@dataclass
class StructureCreateFormat:
    name: str
    timestamp: float
    operator: str
    data: Union[Tuple, List[Tuple]]


def check_structure_create_format(message: StructureCreateFormat):
    assert (
        type(message.name) == str
    ), "the type of StructureCreateFormat.name must be string"
    assert (
        type(message.timestamp) == float
    ), "the type of StructureCreateFormat.name must be float"
    assert (
        type(message.operator) == str
    ), "the type of StructureCreateFormat.operation must be string"
    assert (
        type(message.data) == tuple or type(message.data) == list
    ), "the type of StructureCreateFormat.data must be Union[Tuple, List[Tuple]]"


@dataclass
class StructureRetrieveFormat:
    name: str
    timestamp: float
    rank_order: int = 1


def check_structure_retrieve_format(message: StructureRetrieveFormat):
    assert (
        type(message.name) == str
    ), "the type of StructureRetrieveFormat.name must be string"
    assert (
        type(message.timestamp) == float
    ), "the type of StructureRetrieveFormat.name must be float"
    assert (
        type(message.rank_order) == int
    ), "the type of StructureCreateFormat.rank_order must int"


@dataclass
class StructureDeleteFormat:
    name: str
    timestamp: float


def check_structure_delete_format(message: StructureDeleteFormat):
    assert (
        type(message.name) == str
    ), "the type of StructureDeleteFormat.name must be string"
    assert (
        type(message.timestamp) == float
    ), "the type of StructureDeleteFormat.name must be float"
