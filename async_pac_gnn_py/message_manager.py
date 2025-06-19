#!/usr/bin/env python3

import torch
import numpy as np
from async_pac_gnn_interfaces.msg import RobotPositions, Features
from std_msgs.msg import Header
import coverage_control
import rclpy
from . import utils


class MessageManager:
    """
    Utility class for LPAC distributed processing
    """

    def __init__(self, comm_range: float, ns: str, ns_index: int, world_map_size: int,
                 node=None, qos_profile=None, robot_namespaces=None):
        """
        Initialize MessageManager for distributed processing

        Args:
            comm_range: Communication range for filtering neighbors
            ns: Namespace of this robot
            ns_index: Index of this robot in the pose array
            world_map_size: Size of the world map for position normalization
            node: ROS2 node for creating subscriptions (optional)
            robot_namespaces: List of robot namespaces for CNN feature subscriptions (optional)
        """
        self.comm_range = comm_range
        self.comm_range_sqr = comm_range**2
        self.ns = ns
        self.ns_index = ns_index
        self.world_map_size = world_map_size
        self.half_world_map_size = world_map_size/2.0
        self.qos_profile = qos_profile

        self.relative_neighbors = torch.empty(0, 2, dtype=torch.float32)
        self.edge_index = torch.empty(2, 0, dtype=torch.int64)
        self.all_cnn_features = None

        self.cnn_features_by_ns = {}  # {namespace: Features message}
        self.last_position_update_time = None

        self.cnn_features_pub = None
        self.own_cnn_features = None

        self.sim_pos_tensor = None

        if node is not None and robot_namespaces is not None:
            self._setup_pub_sub(node, robot_namespaces)

    def _setup_pub_sub(self, node, robot_namespaces):
        """
        Setup ROS2 publisher for own_cnn_features
        Setup ROS2 subscription for other cnn_features

        Args:
            node: ROS2 node
            robot_namespaces: List of robot namespaces
        """

        self.cnn_features_pub = node.create_publisher(
                Features,
                'cnn_features',
                self.qos_profile
                )

        # Subscribe to CNN features from other robots (exclude own namespace)
        self.cnn_feature_subscriptions = {}
        for robot_ns in robot_namespaces:
            if robot_ns != self.ns:  # Don't subscribe to own features
                topic_name = f'/{robot_ns}/cnn_features'
                subscription = node.create_subscription(
                        Features,
                        topic_name,
                        lambda msg, ns=robot_ns: self._cnn_features_callback(msg, ns),
                        self.qos_profile
                        )
                self.cnn_feature_subscriptions[robot_ns] = subscription

    def _cnn_features_callback(self, msg: Features, robot_ns: str):
        """Store latest CNN features message for each robot if timestamp is recent"""
        if self.last_position_update_time is not None:
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            if msg_time < self.last_position_update_time:
                return

        self.cnn_features_by_ns[robot_ns] = msg

    def process_cnn_features(self):
        """
        Process CNN features from all robots to extract positions and create complete graph structure
        """
        self.last_position_update_time = rclpy.clock.Clock().now()

        num_robots = len(self.cnn_features_by_ns) + 1  # +1 for self

        all_robot_positions = torch.zeros((num_robots, 2), dtype=torch.float32)
        all_robot_positions[0] = self.sim_pos_tensor

        feature_dim = self.own_cnn_features.shape[0]
        all_cnn_features = torch.zeros((num_robots, feature_dim), dtype=torch.float32)

        all_cnn_features[0] = self.own_cnn_features

        for i, (robot_ns, features_msg) in enumerate(self.cnn_features_by_ns.items(), 1):
            all_cnn_features[i] = utils.features_to_tensor(features_msg)
            all_robot_positions[i] = all_cnn_features[i][-1:] * self.world_map_size - self.half_world_map_size

        self.all_cnn_features = all_cnn_features

        adjacency_matrix = torch.cdist(all_robot_positions, all_robot_positions, 2) < self.comm_range  # Create adjacency based on comm_range
        adjacency_matrix.fill_diagonal_(0)  # Remove self-loops

        self.edge_index = adjacency_matrix.to_sparse().coalesce().indices().long()

        self.relative_neighbors = torch.empty(0, 2, dtype=torch.float32)
        if num_robots > 1:
            neighbor_mask = adjacency_matrix[0]

            if neighbor_mask.any():
                self.relative_neighbors = all_robot_positions[neighbor_mask] - self.sim_pos_tensor

        self.cnn_features_by_ns = {}


    def get_relative_neighbors_positions_cached(self):
        """
        Get cached relative neighbor positions

        Returns:
            tuple: (relative_neighbors_positions, own_position_xy)
        """
        if self.relative_neighbors is None:
            raise ValueError("Must call process_cnn_features first")
        return self.relative_neighbors, self.sim_pos_tensor

    def get_own_position(self):
        """
        Get cached own position

        Returns:
            torch.Tensor: Own position [x, y]
        """
        # sim_pos_tensor is always available after initialization
        return self.sim_pos_tensor

    def publish_cnn_features(self, cnn_output: torch.Tensor, sim_pos: coverage_control.Point2, frame_id: str = "map"):
        """
        Publish CNN output tensor concatenated with normalized sim position

        Args:
            cnn_output: CNN output tensor
            sim_pos: Simulation position from coverage control environment
            frame_id: Frame ID for the header
        """
        self.sim_pos_tensor = torch.tensor(sim_pos, dtype=torch.float32)

        # Normalize simulation position using same formula as CoverageEnvUtils.normalize_robot_positions
        normalized_position = (self.sim_pos_tensor + self.half_world_map_size) / self.world_map_size

        # Concatenate CNN output with normalized position and store
        self.own_cnn_features = torch.cat([cnn_output.flatten(), normalized_position])

        # Convert to Features message
        msg = Features()
        msg.header = Header()
        msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        msg.header.frame_id = frame_id

        # Flatten tensor and convert to float32 list
        flattened = self.own_cnn_features.detach().cpu().numpy().astype(np.float32)
        msg.features = flattened.tolist()

        # Publish the message
        self.cnn_features_pub.publish(msg)

    def get_all_cnn_features(self):
        """
        Get all CNN features from other robots

        Returns:
            dict: Dictionary of {robot_namespace: Features message} for all robots
        """
        return self.cnn_features_by_ns

    @staticmethod
    def create_manager(comm_range: float, ns: str, ns_index: int, world_map_size: int,
                       node=None, robot_namespaces=None):
        """
        Static factory method to create MessageManager instance

        Args:
            comm_range: Communication range for filtering neighbors
            ns: Namespace of this robot
            ns_index: Index of this robot in the pose array
            world_map_size: Size of the world map for position normalization
            node: ROS2 node for creating subscriptions (optional)
            robot_namespaces: List of robot namespaces for CNN feature subscriptions (optional)

        Returns:
            MessageManager: Configured manager instance
        """
        return MessageManager(comm_range, ns, ns_index, world_map_size, node, robot_namespaces)
