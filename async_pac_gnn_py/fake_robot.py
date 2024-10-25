import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

class FakeRobot(Node):
    def __init__(self):
        super().__init__('fake_robot')

        self.declare_parameter('buffer_size', 10)
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value

        self.declare_parameter('pos_x', 0.0)
        self.init_pos_x = self.get_parameter('pos_x').get_parameter_value().double_value

        self.declare_parameter('pos_y', 0.0)
        self.init_pos_y = self.get_parameter('pos_y').get_parameter_value().double_value

        self.current_pos = PoseStamped()
        self.current_pos.pose.position.x = self.init_pos_x
        self.current_pos.pose.position.y = self.init_pos_y
        self.current_pos.pose.position.z = 0.0
        self.current_pos.pose.orientation.x = 0.0
        self.current_pos.pose.orientation.y = 0.0
        self.current_pos.pose.orientation.z = 0.0
        self.current_pos.pose.orientation.w = 1.0

        self.current_vel = TwistStamped()
        self.current_vel.twist.linear.x = 0.0
        self.current_vel.twist.linear.y = 0.0
        self.current_vel.twist.linear.z = 0.0
        self.current_vel.twist.angular.x = 0.0
        self.current_vel.twist.angular.y = 0.0
        self.current_vel.twist.angular.z = 0.0
        self.last_vel_update_time = self.get_clock().now()

        self.declare_parameter('speed_limit', 1.0)
        self.speed_limit = self.get_parameter('speed_limit').get_parameter_value().double_value

        self.subscription = self.create_subscription(
                TwistStamped,
                'cmd_vel',
                self.cmd_vel_callback,
                10)

        # self.publishers_cmd_vel = [self.create_publisher(TwistStamped, f'{ns}/{cmd_vel_topic}', 10) for ns in self.namespaces_of_robots]
        self.publisher_pose = self.create_publisher(PoseStamped, 'pose', 10)
        self.timer = self.create_timer(0.030, self.timer_callback)

    def cmd_vel_callback(self, msg):
        vel_x = msg.twist.linear.x
        vel_y = msg.twist.linear.y
        vel = np.sqrt(vel_x**2 + vel_y**2)
        if vel > self.speed_limit:
            vel_x = self.speed_limit * vel_x / vel
            vel_y = self.speed_limit * vel_y / vel
        self.current_vel.twist.linear.x = vel_x
        self.current_vel.twist.linear.y = vel_y
        # self.last_vel_update_time = self.get_clock().now()
        # self.current_pos[0] += vel_x
        # self.current_pos[1] += vel_y
        # show current pos and velocity as ros info
        # self.get_logger().info(f'current_pos: {self.current_pos}, vel: {vel_x, vel_y}')

    def timer_callback(self):
        # self.current_pos[0] += self.current_vel.twist.linear.x * 0.03
        # self.current_pos[1] += self.current_vel.twist.linear.y * 0.03
        self.current_pos.header.stamp = self.get_clock().now().to_msg()
        self.current_pos.pose.position.x += self.current_vel.twist.linear.x * 0.03
        self.current_pos.pose.position.y += self.current_vel.twist.linear.y * 0.03
        self.publisher_pose.publish(self.current_pos)


def main(args=None):
    rclpy.init(args=args)
    fake_robot_node = FakeRobot()
    rclpy.spin(fake_robot_node)
    fake_robot_node.destroy_node()
    rclpy.shutdown()
