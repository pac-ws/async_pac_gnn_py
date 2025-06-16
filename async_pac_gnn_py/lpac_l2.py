from __future__ import annotations
import os
import numpy as np
import torch

import rclpy
from geometry_msgs.msg import TwistStamped
from async_pac_gnn_interfaces.msg import Features

import coverage_control

from async_pac_gnn_py.lpac_abstract import LPACAbstract
from async_pac_gnn_py.lpac_cnn_backbone import LPACCNNBackbone
from async_pac_gnn_py.lpac_gnn_backbone import LPACGNNBackbone
from async_pac_gnn_py.message_manager import MessageManager, features_to_tensor


class LPAC_L2(LPACAbstract):
    def __init__(self):
        super().__init__('lpac_l2_distributed', is_solo=True)

        self._get_namespace_index()
        self._create_cmd_vel_publishers(is_solo=True)

        # Initialize MessageManager with subscriptions
        self.message_manager = MessageManager(
            comm_range=self._cc_parameters.pCommunicationRange,
            ns=self._ns,
            ns_index=self._ns_index,
            world_map_size=self._cc_parameters.pWorldMapSize,
            node=self,
            robot_namespaces=self._ns_robots
        )

        # Create single-robot coverage control environment
        # Use own position as initial position
        initial_position = coverage_control.PointVector([[0.0, 0.0]])
        if not self._create_cc_env(self._idf_path, initial_position):
            raise RuntimeError('Coverage control environment creation failed')

        # Initialize CNN and GNN backbones
        self.cnn_backbone = LPACCNNBackbone(self._controller_params, self._cc_parameters)
        self.gnn_backbone = LPACGNNBackbone(self._controller_params, self._cc_parameters)

        # Publishers
        self.cnn_features_pub = self.create_publisher(
            Features,
            f'/{self._ns}/cnn_features',
            self._qos_profile
        )

        # Timers
        cnn_timer_period = self._cc_parameters.pTimeStep  # CNN processing frequency
        gnn_timer_period = self._cc_parameters.pTimeStep * 2  # GNN processing at half frequency
        
        self.cnn_timer = self.create_timer(cnn_timer_period, self._cnn_processing_callback)
        self.gnn_timer = self.create_timer(gnn_timer_period, self._gnn_processing_callback)

        # Pre-allocate reusable message objects
        self._twist_msg = TwistStamped()
        self.cnn_features = None

        self.get_logger().info(f'LPAC_L2 node initialized for robot {self._ns} {self._ns_index}/{self._num_robots}')

    def _cnn_processing_callback(self):
        """CNN processing timer callback"""
        if self._status_pac not in [0, 1] or self._cc_env is None:
            return

        try:
            # Get cached relative neighbor positions
            if (self.message_manager.relative_neighbors is not None and 
                self.message_manager.own_position is not None):
                
                # Execute CNN forward pass with available data
                cnn_output = self.cnn_backbone.step(self._cc_env, self.message_manager.relative_neighbors)
                
                # Update CNN features with normalized position
                self.cnn_features = self.message_manager.cnn_output_to_features(cnn_output)
                
                # Publish CNN features
                self.cnn_features_pub.publish(self.cnn_features)
            
            # Process robot positions at end to prepare for next cycle
            self.message_manager.process_robot_positions()
            
        except Exception as e:
            self.get_logger().error(f'CNN processing error: {e}')

    def _gnn_processing_callback(self):
        """GNN processing timer callback"""
        if self._status_pac not in [0, 1]:
            return

        try:
            # Get neighbors CNN features
            neighbors_features = self.message_manager.get_neighbors_cnn_features()
            edge_index = self.message_manager.get_edge_index()
            
            if self.cnn_features is None:
                self.get_logger().warn('No CNN features available for GNN processing')
                return

            # Construct node feature matrix
            node_features = self._construct_node_features(neighbors_features)
            
            if node_features is not None and edge_index.numel() > 0:
                # Execute GNN forward pass
                gnn_output = self.gnn_backbone.step(node_features, edge_index)
                
                # Convert GNN output to velocity command
                velocity = self._gnn_output_to_velocity(gnn_output)
                
                # Publish velocity command
                self._publish_velocity_command(velocity)
            else:
                # No neighbors - publish zero velocity
                self._publish_velocity_command([0.0, 0.0])
                
        except Exception as e:
            self.get_logger().error(f'GNN processing error: {e}')

    def _construct_node_features(self, neighbors_features: dict) -> torch.Tensor:
        """
        Construct node feature matrix for GNN input
        
        Args:
            neighbors_features: Dictionary of neighbor CNN features
            
        Returns:
            torch.Tensor: Node features [num_nodes, feature_dim] or None if no features
        """
        if self.cnn_features is None:
            return None

        # Convert own CNN features to tensor
        own_features = features_to_tensor(self.cnn_features)
        
        # Collect neighbor features
        neighbor_feature_list = []
        for robot_ns, features_msg in neighbors_features.items():
            if robot_ns != self._ns:  # Skip self
                neighbor_features = features_to_tensor(features_msg)
                neighbor_feature_list.append(neighbor_features)
        
        # Stack features: self first, then neighbors
        if len(neighbor_feature_list) == 0:
            # Only self
            node_features = own_features.unsqueeze(0)  # [1, feature_dim]
        else:
            # Self + neighbors
            all_features = [own_features] + neighbor_feature_list
            node_features = torch.stack(all_features, dim=0)  # [num_nodes, feature_dim]
        
        return node_features

    def _gnn_output_to_velocity(self, gnn_output: torch.Tensor) -> list:
        """
        Convert GNN output to velocity command
        
        Args:
            gnn_output: GNN output tensor for this robot
            
        Returns:
            list: [vx, vy] velocity command
        """
        # Convert tensor to numpy and extract velocity
        velocity_np = gnn_output.detach().cpu().numpy().flatten()
        
        # Take first two elements as velocity command
        if len(velocity_np) >= 2:
            return [float(velocity_np[0]), float(velocity_np[1])]
        else:
            return [0.0, 0.0]

    def _publish_velocity_command(self, velocity: list):
        """
        Publish velocity command
        
        Args:
            velocity: [vx, vy] velocity command
        """
        # Reuse pre-allocated message object
        t = self.get_clock().now()
        self._twist_msg.header.stamp = t.to_msg()
        self._twist_msg.header.frame_id = self._ns
        self._twist_msg.twist.linear.x = float(velocity[0] * self._vel_scale)
        self._twist_msg.twist.linear.y = float(velocity[1] * self._vel_scale)

        self._cmd_vel_publishers[0].publish(self._twist_msg)

    def destroy_node(self):
        """Clean shutdown of the node."""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    lpac_node = LPAC_L2()
    executor = rclpy.executors.SingleThreadedExecutor()
    try:
        executor.add_node(lpac_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        lpac_node.get_logger().error(f'Exception occurred in spinning: {e}')
    finally:
        lpac_node.get_logger().info('Shutting down LPAC_L2 node')
        executor.remove_node(lpac_node)
        executor.shutdown()
        lpac_node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()