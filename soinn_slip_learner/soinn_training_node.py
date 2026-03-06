#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from soinn_py import SoinnPlus


class SoinnTrainingNode(Node):
    def __init__(self) -> None:
        super().__init__('soinn_training_node')
        self.declare_parameter('model_path', 'models/soinn_model.pkl')
        self.declare_parameter('init_new_model', True)
        self.declare_parameter('sample_topic', '/experience_samples')
        self.declare_parameter('input_dimension', 0)
        self.declare_parameter('max_edge_age', 50)
        self.declare_parameter('auto_save_period_sec', 60.0)

        self.model_path = self.get_parameter('model_path').value
        self.init_new_model = self.get_parameter('init_new_model').value
        self.sample_topic = self.get_parameter('sample_topic').value
        self.input_dimension = int(self.get_parameter('input_dimension').value)
        self.max_edge_age = int(self.get_parameter('max_edge_age').value)
        self.auto_save_period_sec = float(self.get_parameter('auto_save_period_sec').value)

        self.training_samples = 0
        self.model_dir = Path(os.path.dirname(self.model_path)) if os.path.dirname(self.model_path) else None
        if self.model_dir is not None:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.soinn = self._initialize_model()

        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.sample_topic,
            self._experience_callback,
            10,
        )

        self.save_timer = self.create_timer(self.auto_save_period_sec, self._save_model)
        self.get_logger().info(
            f'soinn_training_node started, topic={self.sample_topic}, model_path={self.model_path}'
        )

    def _initialize_model(self) -> SoinnPlus:
        if (not self.init_new_model) and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as file:
                    model = pickle.load(file)
                self.get_logger().info(f'Loaded existing model from {self.model_path}')
                return model
            except Exception as error:
                self.get_logger().warn(
                    f'Failed to load model from {self.model_path}: {error}. Initializing new model.'
                )

        if self.input_dimension <= 0:
            self.get_logger().warn(
                'input_dimension <= 0, deferring model initialization until first sample arrives'
            )
            return None

        self.get_logger().info(
            f'Initializing new SOINN model (dim={self.input_dimension}, max_edge_age={self.max_edge_age})'
        )
        return SoinnPlus(dim=self.input_dimension, age_max=self.max_edge_age)

    def _experience_callback(self, msg: Float32MultiArray) -> None:
        sample = np.array(msg.data, dtype=float)
        if sample.size == 0:
            self.get_logger().warn('Received empty experience sample; skipping')
            return

        if self.soinn is None:
            inferred_dim = int(sample.size)
            self.soinn = SoinnPlus(dim=inferred_dim, age_max=self.max_edge_age)
            self.input_dimension = inferred_dim
            self.get_logger().info(f'Initialized SOINN model from first sample with dim={inferred_dim}')

        if sample.size != self.soinn.dimension:
            self.get_logger().warn(
                f'Received sample with size={sample.size}, expected={self.soinn.dimension}; skipping'
            )
            return

        try:
            self.soinn.input_signal(sample)
            self.training_samples += 1
            if self.training_samples % 100 == 0:
                self.get_logger().info(f'Trained on {self.training_samples} samples')
        except Exception as error:
            self.get_logger().error(f'Error while training on incoming sample: {error}')

    def _save_model(self) -> None:
        if self.soinn is None:
            return
        try:
            tmp_path = f'{self.model_path}.tmp'
            with open(tmp_path, 'wb') as file:
                pickle.dump(self.soinn, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, self.model_path)
            self.get_logger().debug(f'Model saved to {self.model_path}')
        except Exception as error:
            self.get_logger().error(f'Could not save model to {self.model_path}: {error}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SoinnTrainingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down training node')
    finally:
        node._save_model()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
