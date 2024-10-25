import os
import sys
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

def main():
    # subscribe to /robot_0/local_map topic which is of type std_msgs/Float32MultiArray
    rclpy.init(args=sys.argv)
    node = Node('local_map_subscriber')
    node.create_subscription(Float32MultiArray, '/robot_1/local_map', callback, 10)
    rclpy.spin(node)
    rclpy.shutdown()



def callback(data):
    # data.data is a list of floats
    # convert to numpy array
    local_map = np.array(data.data, dtype=np.float32)
    # reshape to 2D array
    local_map = local_map.reshape(data.layout.dim[0].size, data.layout.dim[1].size)
    # convert to torch tensor
    local_map = torch.from_numpy(local_map)
    # save to file
    print("Saving local map to local_map.pt")
    print("Sum of local map: ", torch.sum(local_map).item())
    torch.save(local_map, 'local_map.pt')

if __name__ == '__main__':
    main()
