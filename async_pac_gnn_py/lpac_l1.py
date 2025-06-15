from __future__ import annotations
import os
import numpy as np

import rclpy
from rclpy.wait_for_message import wait_for_message
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import TwistStamped

import coverage_control

from async_pac_gnn_py.lpac_controller import LPACController
from async_pac_gnn_py.lpac_abstract import LPACAbstract

class LPAC(LPACAbstract):
    def __init__(self):
        super().__init__('lpac_coveragecontrol', is_solo=False)

        self._get_namespace_index()
        self._create_cmd_vel_publishers(is_solo=True)

        self._robot_poses_topic = 'robot_poses'
        self._wait_for_robot_poses()
        self._initialize_poses_subscription()

        if not self._create_cc_env(self._idf_path, self._robot_poses):
            raise RuntimeError('Coverage control environment creation failed')

        timer_period = self._cc_parameters.pTimeStep
        self.lpac_step_timer = self.create_timer(timer_period, self._lpac_step_callback)

        if self._cc_env is not None:
            self.controller = LPACController(
                    self._controller_params, self._cc_parameters)
        else:
            raise RuntimeError('Coverage control environment is None')
        
        # Pre-allocate reusable message object to avoid repeated creation
        self._twist_msg = TwistStamped()
        
        self.get_logger().info(f'LPAC node initialized for robot {self._ns} {self._ns_index}/{self._num_robots}')

    def _initialize_poses_subscription(self):
        self._poses_subscription = self.create_subscription(
                PoseArray,
                self._robot_poses_topic,
                self._poses_callback,
                qos_profile=self._qos_profile)


    def _robot_poses_from_msg(self, msg: PoseArray) -> coverage_control.PointVector:
        return coverage_control.PointVector(
            [[pose.position.x, pose.position.y] for pose in msg.poses]
        )
    def _wait_for_robot_poses(self):
        while True:
            ok, msg = wait_for_message(
                PoseArray,
                self,
                self._robot_poses_topic,
                qos_profile=self._qos_profile,
                time_to_wait=5.0)
            if ok:
                self._robot_poses = self._robot_poses_from_msg(msg)
                break
            self.get_logger().warn('Waiting for robot poses...', once=True)

    def _poses_callback(self, msg):
            self._robot_poses = self._robot_poses_from_msg(msg)
            if self._cc_env is not None and self._status_pac == 0:
                self._cc_env.SetGlobalRobotPositions(self._robot_poses)

    def _lpac_step_callback(self):
        if self._status_pac not in [0, 1]:
            return
        self_action = [0.0, 0.0]
        if self._status_pac == 0 and self._cc_env is not None:
            actions = self.controller.step(self._cc_env)
            self_action = actions[self._ns_index]

        # Reuse pre-allocated message object
        t = self.get_clock().now()
        self._twist_msg.header.stamp = t.to_msg()
        self._twist_msg.header.frame_id = self._ns
        self._twist_msg.twist.linear.x = float(self_action[0] * self._vel_scale)
        self._twist_msg.twist.linear.y = float(self_action[1] * self._vel_scale)

        self._cmd_vel_publishers[0].publish(self._twist_msg)

    def float32_multiarray_to_numpy(self, msg: Float32MultiArray) -> np.ndarray:
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size
        matrix = np.array(msg.data, dtype=np.float32).reshape((rows, cols))
        return matrix

    def destroy_node(self):
        """Clean shutdown of the node."""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    lpac_node = LPAC()
    executor = rclpy.executors.SingleThreadedExecutor()
    try:
        executor.add_node(lpac_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        lpac_node.get_logger().error(f'Exception occurred in spinning: {e}')
    finally:
        lpac_node.get_logger().info('Shutting down LPAC node')
        executor.remove_node(lpac_node)
        executor.shutdown()
        lpac_node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()
