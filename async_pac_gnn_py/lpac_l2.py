from __future__ import annotations
import os
import numpy as np
import torch

import rclpy
from rclpy.wait_for_message import wait_for_message
from geometry_msgs.msg import TwistStamped, PoseStamped
from async_pac_gnn_interfaces.msg import Features

import coverage_control
from coverage_control import Point2
from coverage_control import IOUtils

from async_pac_gnn_py.lpac_abstract import LPACAbstract
from async_pac_gnn_py.cnn_backbone import LPACCNNBackbone
from async_pac_gnn_py.gnn_backbone import LPACGNNBackbone
from async_pac_gnn_py.message_manager import MessageManager

from . import utils

class LPAC_L2(LPACAbstract):
    def __init__(self):
        super().__init__('lpac_l2_distributed', is_solo=True)

        self._get_namespace_index()
        self._create_cmd_vel_publishers(is_solo=True)

        self.sim_pos = None
        self.world_xy = None

        self._robot_pose_topic = 'pose'
        self._wait_for_robot_pose()
        self._initialize_pose_subscription()

        initial_position = coverage_control.PointVector([self.sim_pos])
        if not self._create_cc_env(self._idf_path, initial_position):
            raise RuntimeError('Coverage control environment creation failed')
        self.sim_pos = self._cc_env.GetRobotPosition(0)

        self.message_manager = MessageManager(
                comm_range=self._cc_parameters.pCommunicationRange,
                ns=self._ns,
                ns_index=self._ns_index,
                world_map_size=self._cc_parameters.pWorldMapSize,
                node=self,
                qos_profile=self._qos_profile,
                robot_namespaces=self._ns_robots
                )

        lparams = IOUtils.load_toml(IOUtils.sanitize_path(self._controller_params["LearningParams"]))
        model_state_dicts = utils.load_and_split_state_dict(self._controller_params["ModelStateDict"])
        self.cnn_backbone = LPACCNNBackbone(self._controller_params, self._cc_parameters, model_state_dicts)
        self.gnn_backbone = LPACGNNBackbone(self._controller_params, self._cc_parameters, model_state_dicts)

        timer_period = self._cc_parameters.pTimeStep
        self.cnn_timer = self.create_timer(timer_period, self._cnn_processing_callback)
        self.gnn_timer = self.create_timer(timer_period, self._gnn_processing_callback)
        self.position_timer = self.create_timer(timer_period, self._position_update_callback)

        # Pre-allocate reusable message objects
        self._twist_msg = TwistStamped()
        self.latest_cnn_output = None

        self.get_logger().info(f'LPAC_L2 node initialized for robot {self._ns} {self._ns_index}/{self._num_robots}')

    def _initialize_pose_subscription(self):
        self.pose_subscription = self.create_subscription(
                PoseStamped,
                self._robot_pose_topic,
                self._pose_callback,
                qos_profile=self._qos_profile,
                )

    def _pose_callback(self, msg: PoseStamped):
        """Store latest pose message"""
        self.world_xy = coverage_control.Point2(
                msg.pose.position.x,
                msg.pose.position.y
                )

    def _wait_for_robot_pose(self):
        while True:
            ok, msg = wait_for_message(
                    PoseStamped,
                    self,
                    self._robot_pose_topic,
                    qos_profile=self._qos_profile,
                    time_to_wait=5.0)
            if ok:
                self._pose_callback(msg)
                self.sim_pos = self.world_xy * self._env_scale
                break
            self.get_logger().warn('Waiting for robot poses...', once=True)

    def _position_update_callback(self):
        """Position update timer callback - 500ms"""
        if self._cc_env is not None and self._status_pac == 0:
            try:
                # Scale world position by env_scale and update coverage control environment
                scaled_pos = Point2(
                        self.world_xy[0] * self._env_scale,
                        self.world_xy[1] * self._env_scale
                        )
                self._cc_env.SetGlobalRobotPosition(0, scaled_pos)

                # Get simulation position from environment and store as data member
                self.sim_pos = self._cc_env.GetRobotPosition(0)

            except Exception as e:
                self.get_logger().error(f'Position update error: {e}')

    def _cnn_processing_callback(self):
        """CNN processing timer callback"""
        if self._status_pac not in [0, 1] or self._cc_env is None:
            return

        try:
            # Get cached relative neighbor positions (may be empty initially)
            relative_neighbors = self.message_manager.relative_neighbors
            if relative_neighbors is None:
                relative_neighbors = torch.empty(0, 2, dtype=torch.float32)

            # Execute CNN forward pass with available data
            cnn_output = self.cnn_backbone.forward(self._cc_env, relative_neighbors)

            # Store CNN output for GNN processing
            self.latest_cnn_output = cnn_output

            # Publish CNN features using MessageManager with sim_pos
            self.message_manager.publish_cnn_features(cnn_output, self.sim_pos)

            # Process CNN features at end to prepare for next cycle
            self.message_manager.process_cnn_features()

        except Exception as e:
            self.get_logger().error(f'CNN processing error: {e}')

    def _gnn_processing_callback(self):
        """GNN processing timer callback"""
        if self._status_pac not in [0, 1]:
            return

        try:
            # Get pre-constructed CNN features and edge index from MessageManager
            node_features = self.message_manager.all_cnn_features
            edge_index = self.message_manager.edge_index

            # Check if we have valid features
            if node_features is None:
                self.get_logger().warn('No CNN features available for GNN processing')
                return

            # Execute GNN forward pass
            gnn_output = self.gnn_backbone.forward(node_features, edge_index)

            self.get_logger().info(f'gnn_output: {gnn_output}')

            # Convert GNN output to velocity command
            velocity = self._gnn_output_to_velocity(gnn_output)

            # Publish velocity command
            self._publish_velocity_command(velocity)

        except Exception as e:
            self.get_logger().error(f'GNN processing error: {e}')


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
