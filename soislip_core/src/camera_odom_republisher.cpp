#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

class CameraOdomRepublisherNode : public rclcpp::Node {
public:
  CameraOdomRepublisherNode()
  : Node("camera_odom_republisher") {
    this->declare_parameter("input_topic", "/camera/odom");
    this->declare_parameter("output_topic", "/camera/odom_republished");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("output_odom_frame", "visual_odom");
    this->declare_parameter("publish_tf", true);

    this->get_parameter("input_topic", input_topic_);
    this->get_parameter("output_topic", output_topic_);
    this->get_parameter("base_frame", base_frame_);
    this->get_parameter("output_odom_frame", output_odom_frame_);
    this->get_parameter("publish_tf", publish_tf_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(output_topic_, 10);
    subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
      input_topic_,
      10,
      std::bind(&CameraOdomRepublisherNode::odom_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "camera_odom_republisher started: input='%s', output='%s', base_frame='%s', output_odom_frame='%s', publish_tf=%s",
      input_topic_.c_str(), output_topic_.c_str(), base_frame_.c_str(), output_odom_frame_.c_str(), publish_tf_ ? "true" : "false");
  }

private:
  bool lookup_child_to_base_transform(
    const std::string & child_frame,
    const rclcpp::Time & stamp,
    geometry_msgs::msg::TransformStamped & child_to_base)
  {
    try {
      child_to_base = tf_buffer_->lookupTransform(base_frame_, child_frame, stamp);
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Failed to lookup transform '%s' <- '%s': %s",
        base_frame_.c_str(),
        child_frame.c_str(),
        ex.what());
    }
    return false;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if (msg->child_frame_id.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Received odometry message with empty child_frame_id on '%s'", input_topic_.c_str());
      return;
    }

    geometry_msgs::msg::TransformStamped child_to_base;
    if (!lookup_child_to_base_transform(msg->child_frame_id, rclcpp::Time(msg->header.stamp), child_to_base)) {
      return;
    }

    nav_msgs::msg::Odometry republished_msg = *msg;

    tf2::Transform odom_to_child;
    tf2::fromMsg(msg->pose.pose, odom_to_child);
    tf2::Transform child_to_base_tf;
    tf2::fromMsg(child_to_base.transform, child_to_base_tf);

    const tf2::Transform odom_to_base = odom_to_child * child_to_base_tf;
    republished_msg.pose.pose.position.x = odom_to_base.getOrigin().x();
    republished_msg.pose.pose.position.y = odom_to_base.getOrigin().y();
    republished_msg.pose.pose.position.z = odom_to_base.getOrigin().z();
    republished_msg.pose.pose.orientation = tf2::toMsg(odom_to_base.getRotation());

    const tf2::Vector3 linear_child(
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y,
      msg->twist.twist.linear.z);
    const tf2::Vector3 angular_child(
      msg->twist.twist.angular.x,
      msg->twist.twist.angular.y,
      msg->twist.twist.angular.z);

    const tf2::Matrix3x3 rotation_child_to_base = child_to_base_tf.getBasis();
    const tf2::Vector3 angular_base = rotation_child_to_base * angular_child;
    const tf2::Vector3 child_origin_in_base = child_to_base_tf.getOrigin();
    const tf2::Vector3 linear_base =
      (rotation_child_to_base * linear_child) - angular_base.cross(child_origin_in_base);

    republished_msg.twist.twist.linear.x = linear_base.x();
    republished_msg.twist.twist.linear.y = linear_base.y();
    republished_msg.twist.twist.linear.z = linear_base.z();
    republished_msg.twist.twist.angular.x = angular_base.x();
    republished_msg.twist.twist.angular.y = angular_base.y();
    republished_msg.twist.twist.angular.z = angular_base.z();

    republished_msg.child_frame_id = base_frame_;
    republished_msg.header.frame_id = output_odom_frame_;

    publisher_->publish(republished_msg);

    if (publish_tf_) {
      if (republished_msg.header.frame_id.empty()) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Cannot publish inverse TF because odometry header.frame_id is empty");
        return;
      }

      geometry_msgs::msg::TransformStamped base_to_odom_tf;
      base_to_odom_tf.header.stamp = republished_msg.header.stamp;
      base_to_odom_tf.header.frame_id = base_frame_;
      base_to_odom_tf.child_frame_id = output_odom_frame_;
      base_to_odom_tf.transform = tf2::toMsg(odom_to_base.inverse());
      tf_broadcaster_->sendTransform(base_to_odom_tf);
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string base_frame_;
  std::string output_odom_frame_;
  bool publish_tf_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraOdomRepublisherNode>());
  rclcpp::shutdown();
  return 0;
}
