import unittest

from experiment.sumo_env import get_node_data, get_edge_data


class TestSumoEnv(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_get_node_data(self):
        step = 0
        type = "test"
        source_list = ["node1", "node2"]

        operation_list = [str]
        # step, node_name, type
        target_node_list = []
        target_node_list.append([0, "node1", "test"])
        target_node_list.append([0, "node2", "test"])
        # step,node_name,operation_results
        target_feat_list = []
        target_feat_list.append([0, "node1", "node1"])
        target_feat_list.append([0, "node2", "node2"])
        node, feat = get_node_data(step, type, source_list, operation_list)

        self.assertEqual(node, target_node_list)
        self.assertEqual(feat, target_feat_list)

    def test_graph_edge_data(self):
        step = 0
        relation = "veh_follow_veh"
        source_list = ["node1", "node2"]
        operation_list = [lambda node_id: f"to_{node_id}"]
        # step, from, to, relation
        target_edge_list = []
        target_edge_list.append([0, "node1", "to_node1", "veh_follow_veh"])
        target_edge_list.append([0, "node2", "to_node2", "veh_follow_veh"])
        edge = get_edge_data(step, relation, source_list, operation_list)
        self.assertEqual(edge, target_edge_list)

    def tearDown(self) -> None:
        return super().tearDown()
