from __future__ import annotations
import os
import numpy as np
import torch
from torch_geometric.nn import MLP

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
from async_pac_gnn_py.lpac_controller import LPACAction

from . import utils

class LPAC_L2(LPACAbstract):
    def __init__(self):
        super().__init__('lpac_l2_distributed', is_solo=True)

        self._create_cmd_vel_publishers(is_solo=True)

        self._sim_poses = coverage_control.PointVector([[0.0, 0.0]])
        self.world_xy = None

        self.cnn_output = None
        self._twist_msg = TwistStamped()

        self._robot_pose_topic = 'pose'

        self.cnn_backbone = None
        self.gnn_backbone = None
        self.lpac_action = None

        self.cnn_timer = None
        self.gnn_timer = None
        self.position_timer = None

        self.message_manager = None

        self.timer_period = self._cc_parameters.pTimeStep

        self.initialize_timer = self.create_timer(1, self.initialize_cb)
        self.state = 0

    def initialize_cb(self):
        if self.state == 0:
            self.InitializeBase()
            self.state = 1
        elif self._base_initialized == False:
            self.InitializeBase()
            return
        elif self.state == 1:
            self._get_namespace_index()
            # self._wait_for_robot_pose()
            self._initialize_pose_subscription()
            self.state = 2
        elif self.state == 2:
            if self.world_xy is None:
                return
            else:
                self.state = 3
        elif self.state == 3:

            self.initialize_timer.cancel()
            if not self._create_cc_env(self._idf_path, self._sim_poses):
                raise RuntimeError('Coverage control environment creation failed')
            self._sim_poses[0] = self._cc_env.GetRobotPosition(0)

            lparams = IOUtils.load_toml(IOUtils.sanitize_path(self._controller_params["LearningParams"]))
            model_state_dicts = utils.load_and_split_state_dict(self._controller_params["ModelStateDict"])

            self.cnn_backbone = LPACCNNBackbone(
                    self._controller_params,
                    self._cc_parameters,
                    model_state_dicts["cnn_backbone"]
                    )
            self.gnn_backbone = LPACGNNBackbone(
                    lparams["GNNBackBone"],
                    model_state_dicts["gnn_backbone"]
                    )

            # TODO DANGER: Why are these hardcoded in coverage_control.nn.models.lpac???
            action_config = {"InputDim": self.gnn_backbone.latent_size,
                             "MLP": { "HiddenUnits": 32, "OutDim": 32 },
                             "OutDim": 2}
            self.lpac_action = LPACAction(
                    action_config,
                    model_state_dicts["action"]
                    )

            self.message_manager = MessageManager(
                    comm_range=self._cc_parameters.pCommunicationRange,
                    ns=self._ns,
                    ns_index=self._ns_index,
                    world_map_size=self._cc_parameters.pWorldMapSize,
                    node=self,
                    qos_profile=self._qos_profile,
                    robot_namespaces=self._ns_robots
                    )

            self.cnn_timer = self.create_timer(self.timer_period, self._cnn_processing_callback)
            self.gnn_timer = self.create_timer(self.timer_period, self._gnn_processing_callback)
            self.position_timer = self.create_timer(self.timer_period, self._position_update_callback)

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
                self._sim_poses[0] = self.world_xy * self._env_scale
                break
            self.get_logger().warn('Waiting for robot poses...', once=True)

    def _position_update_callback(self):
        """Position update timer callback"""
        if self._status_pac == 0:
            scaled_pos = self.world_xy * self._env_scale
            self._cc_env.SetGlobalRobotPosition(0, scaled_pos)
            self._sim_poses[0] = self._cc_env.GetRobotPosition(0)

    def _cnn_processing_callback(self):
        """CNN processing timer callback"""
        if self._status_pac != 0 or self._cc_env is None:
            return

        relative_neighbors = self.message_manager.relative_neighbors

        self.cnn_output = self.cnn_backbone.forward(self._cc_env, relative_neighbors)

        self.message_manager.publish_cnn_features(self.cnn_output, self._sim_poses[0])

        self.message_manager.process_cnn_features()

    def _gnn_processing_callback(self):
        """GNN processing timer callback"""
        if self._status_pac not in [0, 1]:
            return

        node_features = self.message_manager.all_cnn_features
        edge_index = self.message_manager.edge_index

        if node_features is None: # TODO: Change initialization to avoid check?
            self.get_logger().warn('No CNN features available for GNN processing')
            return

        gnn_output = self.gnn_backbone.forward(node_features, edge_index)

        velocity = self.lpac_action.step(gnn_output)
        self._publish_velocity_command(velocity[0])


    def _publish_velocity_command(self, velocity: torch.Tensor):
        """
        Publish velocity command

        Args:
            velocity: [vx, vy] velocity command
        """
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
    executor.add_node(lpac_node)
    executor.spin()
    # try:
    #     executor.add_node(lpac_node)
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # except Exception as e:
    #     lpac_node.get_logger().error(f'Exception occurred in spinning: {e}')
    # finally:
    #     lpac_node.get_logger().info('Shutting down LPAC_L2 node')
    #     executor.remove_node(lpac_node)
    #     executor.shutdown()
    #     lpac_node.destroy_node()
    #     if rclpy.ok():
    #         rclpy.try_shutdown()
