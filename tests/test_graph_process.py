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

    def test_generate_graph_dict(self):
        edge_data = pd.DataFrame(
            {
                "step": [0, 1],
                "from_id": [0, 1],  # veh0, veh1
                "to_id": [1, 2],  # veh1, lane2
                "relation": ["veh_follow_veh", "veh_on_lane"],
            }
        )
        target_graph_dict = {
            ("veh", "follow", "veh"): (
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([0]),
            ),
            ("veh", "on", "lane"): (
                torch.tensor([1]),
                torch.tensor([2]),
                torch.tensor([1]),
            ),
        }
        g_dict = generate_graph_dict(edge_data)
        self.assertEqual(g_dict, target_graph_dict)

    def test_generate_feat_dict(self):
        feat_data = pd.DataFrame(
            {
                "step": [0, 1],
                "name": ["veh0", "veh1"],
                "node_id": [0, 1],
                "feat_length": [5, 6],
            }
        )
        type = "veh"
        target_feat_dict = g_feat_dict = {
            "veh": {
                0: {"feat_length": torch.tensor([5])},
                1: {"feat_length": torch.tensor([6])},
            }
        }
        g_feat_dict = generate_feat_dict(type, feat_data)
        self.assertEqual(g_feat_dict, target_feat_dict)

    def tearDown(self) -> None:
        return super().tearDown()
