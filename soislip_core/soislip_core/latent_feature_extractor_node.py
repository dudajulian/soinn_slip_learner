#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class LatentFeatureExtractorNode(Node):
    def __init__(self) -> None:
        super().__init__('latent_feature_extractor_node')
        self.get_logger().info('latent_feature_extractor_node started')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LatentFeatureExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
