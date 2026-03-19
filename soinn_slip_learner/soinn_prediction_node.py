#!/usr/bin/env python3

import os
import pickle
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from soinn_py import SoinnPlus
from soinn_slip_learner.srv import PredictBatch


class SoinnPredictionNode(Node):
    def __init__(self) -> None:
        super().__init__('soinn_prediction_node')
        self.declare_parameter('model_path', 'models/soinn_model.pkl')
        self.declare_parameter('service_name', 'predict_batch')
        self.declare_parameter('feature_dim', 0)
        self.declare_parameter('model_reload_period_sec', 5.0)

        self.model_path = self.get_parameter('model_path').value
        self.service_name = self.get_parameter('service_name').value
        self.feature_dim = int(self.get_parameter('feature_dim').value)
        self.model_reload_period_sec = float(self.get_parameter('model_reload_period_sec').value)

        self.soinn: Optional[SoinnPlus] = None
        self._last_mtime: Optional[float] = None
        self._try_load_model(force=True)

        self.predict_service = self.create_service(
            PredictBatch,
            self.service_name,
            self._handle_predict_batch,
        )
        self.reload_timer = self.create_timer(self.model_reload_period_sec, self._try_load_model)

        self.get_logger().info(
            f'soinn_prediction_node started, service={self.service_name}, model_path={self.model_path}'
        )

    def _try_load_model(self, force: bool = False) -> None:
        if not os.path.exists(self.model_path):
            if force:
                self.get_logger().warn(f'Model file not found: {self.model_path}')
            return

        try:
            current_mtime = os.path.getmtime(self.model_path)
            if (not force) and (self._last_mtime is not None) and (current_mtime <= self._last_mtime):
                return
        except Exception as error:
            self.get_logger().error(f'Failed to stat model path {self.model_path}: {error}')
            return

        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                with open(self.model_path, 'rb') as file:
                    model = pickle.load(file)

                if not isinstance(model, SoinnPlus):
                    self.get_logger().error(f'Loaded object from {self.model_path} is not a SoinnPlus model')
                    return

                self.soinn = model
                self._last_mtime = current_mtime
                self.get_logger().info(f'Loaded SOINN model from {self.model_path} (dim={self.soinn.dimension})')
                return
            except Exception as error:
                if attempt == attempts:
                    self.get_logger().error(f'Failed to load model from {self.model_path}: {error}')
                    return
                time.sleep(0.02)

    def _handle_predict_batch(
        self,
        request: PredictBatch.Request,
        response: PredictBatch.Response,
    ) -> PredictBatch.Response:
        self._try_load_model()

        if self.soinn is None:
            response.success = Bool(data=False)
            response.message = String(data=f'No model available at {self.model_path}')
            response.predictions = Float32MultiArray(data=[])
            return response

        raw = np.array(request.features.data, dtype=float)
        if raw.size == 0:
            response.success = Bool(data=False)
            response.message = String(data='Empty feature vector provided')
            response.predictions = Float32MultiArray(data=[])
            return response

        requested_feature_dim = int(request.feature_dim)
        if requested_feature_dim > 0:
            feature_dim = requested_feature_dim
        elif self.feature_dim > 0:
            feature_dim = self.feature_dim
        else:
            feature_dim = int(self.soinn.dimension - 1)

        if feature_dim <= 0:
            response.success = Bool(data=False)
            response.message = String(data='Invalid feature dimension; set request.feature_dim or node parameter feature_dim')
            response.predictions = Float32MultiArray(data=[])
            return response

        if raw.size % feature_dim != 0:
            response.success = Bool(data=False)
            response.message = String(
                data=(
                    f'Input feature length {raw.size} is not divisible by feature_dim {feature_dim}'
                )
            )
            response.predictions = Float32MultiArray(data=[])
            return response

        batch = raw.reshape((-1, feature_dim))
        predictions = []
        confidence_scores = []
        for index, signal in enumerate(batch):
            try:
                prediction, confidence = self.soinn.inference(signal)
                if prediction is None:
                    predictions.append(float('nan'))
                    confidence_scores.append(float('nan'))
                else:
                    predictions.append(float(prediction))
                    confidence_scores.append(float(confidence))
            except Exception as error:
                self.get_logger().error(f'Inference failed for sample index {index}: {error}')
                predictions.append(float('nan'))
                confidence_scores.append(float('nan'))

        response.predictions = Float32MultiArray(data=predictions)
        response.confidence_scores = Float32MultiArray(data=confidence_scores)
        response.success = Bool(data=True)
        response.message = String(data=f'Produced {len(predictions)} predictions')
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SoinnPredictionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down prediction node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
