import unittest
import torch
import pandas as pd

from transworld.graph.process import (
    generate_unique_node_id,
    generate_graph_dict,
    generate_feat_dict,
)


class TestGraphProcess(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_generate_unique_node_id(self):
        node_data = pd.DataFrame({"step": [0], "name": ["node1"], "type": ["test"]})
        target_node_id = {"node1": 0}
        node_id = generate_unique_node_id(node_data)
        self.assertEqual(node_id, target_node_id)

    def tearDown(self) -> None:
        return super().tearDown()
