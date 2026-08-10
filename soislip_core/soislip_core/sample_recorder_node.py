#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from soislip_interfaces.msg import SOINNSample


class SampleRecorderNode(Node):
	def __init__(self) -> None:
		super().__init__('sample_recorder_node')

		self.declare_parameter('sample_topic', '/experience_samples')
		self.declare_parameter('output_csv_path', 'experience_samples.csv')

		self.sample_topic = str(self.get_parameter('sample_topic').value)
		self.output_csv_path = str(self.get_parameter('output_csv_path').value)

		output_path = Path(self.output_csv_path)
		if output_path.parent != Path('.'):
			output_path.parent.mkdir(parents=True, exist_ok=True)

		self._csv_file = output_path.open('a', newline='', encoding='ascii')
		self._writer = csv.writer(self._csv_file)

		if output_path.stat().st_size == 0:
			self._writer.writerow(['time', 'x', 'y', 'z', 'a', 'b', 'slope', 'slip'])
			self._csv_file.flush()

		self.subscription = self.create_subscription(
			SOINNSample,
			self.sample_topic,
			self._sample_callback,
			10,
		)

		self._received_samples = 0
		self._written_samples = 0
		self.get_logger().info(
			f'sample_recorder_node started, topic={self.sample_topic}, csv={output_path}'
		)

	def _sample_callback(self, msg: SOINNSample) -> None:
		self._received_samples += 1

		if not msg.has_position:
			return

		if len(msg.features) < 3:
			self.get_logger().warn(
				'Received sample with has_position=true but fewer than 3 features; skipping'
			)
			return

		stamp = msg.position.header.stamp
		timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9

		slip = float(msg.label) if msg.has_label else math.nan

		row = [
			timestamp,
			float(msg.position.point.x),
			float(msg.position.point.y),
			float(msg.position.point.z),
			float(msg.features[0]),
			float(msg.features[1]),
			float(msg.features[2]),
			slip,
		]

		self._writer.writerow(row)
		self._csv_file.flush()
		self._written_samples += 1

		if self._written_samples % 100 == 0:
			self.get_logger().info(
				f'Wrote {self._written_samples} samples to {self.output_csv_path}'
			)

	def destroy_node(self) -> bool:
		try:
			if hasattr(self, '_csv_file') and self._csv_file and not self._csv_file.closed:
				self._csv_file.flush()
				self._csv_file.close()
		finally:
			return super().destroy_node()


def main(args=None) -> None:
	rclpy.init(args=args)
	node = SampleRecorderNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info('Shutting down sample recorder node')
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
