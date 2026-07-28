#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <soislip_interfaces/msg/soinn_sample.hpp>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "soislip_interfaces/srv/get_cell_features.hpp"

enum class OdomSourceType {
  Frame,
  Topic,
};

struct OdomSourceConfig {
  std::string name;
  OdomSourceType configured_type{OdomSourceType::Frame};
};

class RobotExperienceCollectorNode : public rclcpp::Node {
public:
  RobotExperienceCollectorNode()
  : Node("robot_experience_collector_node") {
    this->declare_parameter("wheel_separation", 0.3);
    this->declare_parameter("robot_frame", "base_link");
    this->declare_parameter("wheel_odom", "odom");
    this->declare_parameter("reference_odom", "map");
    this->declare_parameter("wheel_odom_source", "topic");
    this->declare_parameter("reference_odom_source", "topic");
    this->declare_parameter("sample_distance", 0.3);
    this->declare_parameter("min_distance_threshold", 0.05);
    this->declare_parameter("min_velocity_threshold", 0.05);
    this->declare_parameter("min_acceleration_threshold", 0.1);
    this->declare_parameter("odom_timeout_sec", 0.5);
    this->declare_parameter("output_topic", "/experience_samples");
    this->declare_parameter("feature_service_name", "get_cell_features");
    this->declare_parameter("tf_poll_period_sec", 0.05); // 20 Hz like the /tf topic

    std::string wheel_odom_source;
    std::string reference_odom_source;
    this->get_parameter("wheel_separation", wheel_separation_);
    this->get_parameter("robot_frame", robot_frame_);
    this->get_parameter("wheel_odom", wheel_odom_);
    this->get_parameter("reference_odom", reference_odom_);
    this->get_parameter("wheel_odom_source", wheel_odom_source);
    this->get_parameter("reference_odom_source", reference_odom_source);
    this->get_parameter("sample_distance", sample_distance_);
    this->get_parameter("min_distance_threshold", min_distance_threshold_);
    this->get_parameter("min_acceleration_threshold", min_acceleration_threshold_);
    this->get_parameter("odom_timeout_sec", odom_timeout_sec_);
    this->get_parameter("min_velocity_threshold", min_velocity_threshold_);
    this->get_parameter("output_topic", output_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("tf_poll_period_sec", tf_poll_period_sec_);

    wheel_odom_config_.name = wheel_odom_;
    wheel_odom_config_.configured_type = parse_source_type(wheel_odom_source, "wheel_odom_source");
    reference_odom_config_.name = reference_odom_;
    reference_odom_config_.configured_type = parse_source_type(reference_odom_source, "reference_odom_source");

    sample_pub_ = this->create_publisher<soislip_interfaces::msg::SOINNSample>(output_topic_, 10);
    feature_client_ = this->create_client<soislip_interfaces::srv::GetCellFeatures>(feature_service_name_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    initialize_odometry_subscribers();

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(tf_poll_period_sec_),
      std::bind(&RobotExperienceCollectorNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "robot_experience_collector_node started, yay");
  }

private:
  void timer_callback() {
    tf2::Transform current_wheelodom;
    tf2::Transform current_ref;

    if (!get_robot_transforms(current_wheelodom, current_ref)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to get robot transforms (wheel_odom='%s', reference_odom='%s')",
        wheel_odom_.c_str(), reference_odom_.c_str());
      skip_iteration_ = true;
      return;
    }

    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Got robot transforms: wheel_odom=(%.2f, %.2f), reference_odom=(%.2f, %.2f)",
      current_wheelodom.getOrigin().x(), current_wheelodom.getOrigin().y(),
      current_ref.getOrigin().x(), current_ref.getOrigin().y());
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "initialized_=%s, skip_iteration_=%s",
      initialized_ ? "true" : "false", skip_iteration_ ? "true" : "false");

    if (!initialized_) {
      transform_last_wheelodom_ = current_wheelodom;
      transform_last_ref_ = current_ref;
      initialized_ = true;
      return;
    }

    if (skip_iteration_) {
      skip_iteration_ = false;
      transform_last_wheelodom_ = current_wheelodom;
      transform_last_ref_ = current_ref;
      return;
    }

    if (!check_minimum_velocity()) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Minimum velocity check failed");
      return;
    }

    float slip = 0.0F;
    if (!calculate_slip(transform_last_wheelodom_, transform_last_ref_, current_wheelodom, current_ref, slip)) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Slip calculation failed (displacement too small or below threshold)");
      return;
    }

    if (!feature_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000, "Feature service '%s' not available",
        feature_service_name_.c_str());
    } else {
      geometry_msgs::msg::Point midpoint = calculate_midpoint_position(transform_last_ref_, current_ref);
      auto request = std::make_shared<soislip_interfaces::srv::GetCellFeatures::Request>();
      request->position = midpoint;

      feature_client_->async_send_request(
        request,
        [this, slip](rclcpp::Client<soislip_interfaces::srv::GetCellFeatures>::SharedFuture future) {
          try {
            auto response = future.get();
            if (!response->success.data) {
              RCLCPP_WARN(this->get_logger(), "Feature request failed: %s", response->message.data.c_str());
              return;
            }

            soislip_interfaces::msg::SOINNSample sample;
            sample.features.assign(response->features.data.begin(), response->features.data.end());
            sample.label = slip;
            sample.has_label = true;
            RCLCPP_INFO(this->get_logger(), "Publishing experience sample with %zu features and label %.3f", sample.features.size(), sample.label);
            sample_pub_->publish(sample);
          } catch (const std::exception & ex) {
            RCLCPP_WARN(this->get_logger(), "Feature service response failed: %s", ex.what());
          }
        });
    }

    transform_last_wheelodom_ = current_wheelodom;
    transform_last_ref_ = current_ref;
  }

  OdomSourceType parse_source_type(const std::string & value, const std::string & parameter_name) {
    if (value == "frame") {
      return OdomSourceType::Frame;
    }
    if (value == "topic") {
      return OdomSourceType::Topic;
    }

    RCLCPP_ERROR(
      this->get_logger(),
      "Unknown value '%s' for parameter '%s'. Expected 'frame' or 'topic'.",
      value.c_str(), parameter_name.c_str());
    throw std::invalid_argument("Unknown odometry source type: " + value);
  }

  void initialize_odometry_subscribers() {
    if (wheel_odom_config_.configured_type == OdomSourceType::Topic) {
      wheel_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        wheel_odom_config_.name,
        10,
        std::bind(&RobotExperienceCollectorNode::wheel_odom_callback, this, std::placeholders::_1));
    }

    if (reference_odom_config_.configured_type == OdomSourceType::Topic) {
      reference_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        reference_odom_config_.name,
        10,
        std::bind(&RobotExperienceCollectorNode::reference_odom_callback, this, std::placeholders::_1));
    }
  }

  void wheel_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    latest_wheel_odom_msg_ = msg;
  }

  void reference_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    latest_reference_odom_msg_ = msg;
  }

  geometry_msgs::msg::Point calculate_midpoint_position(
    const tf2::Transform & transform1_ref,
    const tf2::Transform & transform2_ref) const
  {
    const tf2::Vector3 p1 = transform1_ref.inverse().getOrigin();
    const tf2::Vector3 p2 = transform2_ref.inverse().getOrigin();
    const tf2::Vector3 mid = 0.5 * (p1 + p2);

    geometry_msgs::msg::Point point;
    point.x = mid.x();
    point.y = mid.y();
    point.z = mid.z();
    return point;
  }

  bool get_transform_from_topic(
    tf2::Transform & transform,
    const nav_msgs::msg::Odometry::SharedPtr & odom_msg,
    const std::string & source_name)
  {
    if (!odom_msg) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for Odometry messages on '%s'",
        source_name.c_str());
      return false;
    }

    const rclcpp::Time msg_stamp(odom_msg->header.stamp);
    const rclcpp::Duration timeout = rclcpp::Duration::from_seconds(odom_timeout_sec_);
    if (msg_stamp.nanoseconds() > 0 && (this->now() - msg_stamp) > timeout) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Odometry topic '%s' timed out. Latest message age exceeds %.3f s.",
        source_name.c_str(), odom_timeout_sec_);
      return false;
    }

    if (!odom_msg->child_frame_id.empty() && odom_msg->child_frame_id != robot_frame_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Odometry topic '%s' uses child_frame_id '%s' but robot_frame is '%s'.",
        source_name.c_str(), odom_msg->child_frame_id.c_str(), robot_frame_.c_str());
      return false;
    }

    tf2::fromMsg(odom_msg->pose.pose, transform);
    return true;
  }

  bool get_transform_from_frame(
    tf2::Transform & transform,
    const std::string & source_frame,
    const tf2::TimePoint & t = tf2::TimePointZero)
  {
    try {
      tf2::fromMsg(tf_buffer_->lookupTransform(robot_frame_, source_frame, t).transform, transform);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Could not get transform from frame '%s': %s",
        source_frame.c_str(), ex.what());
      return false;
    }
    return true;
  }

  bool get_single_transform(
    tf2::Transform & transform,
    const OdomSourceConfig & config,
    const nav_msgs::msg::Odometry::SharedPtr & odom_msg,
    const tf2::TimePoint & t = tf2::TimePointZero)
  {
    if (config.configured_type == OdomSourceType::Topic) {
      return get_transform_from_topic(transform, odom_msg, config.name);
    }
    return get_transform_from_frame(transform, config.name, t);
  }

  bool get_robot_transforms(
    tf2::Transform & transform_wheelodom,
    tf2::Transform & transform_ref,
    const tf2::TimePoint & t = tf2::TimePointZero)
  {
    return
      get_single_transform(transform_wheelodom, wheel_odom_config_, latest_wheel_odom_msg_, t) &&
      get_single_transform(transform_ref, reference_odom_config_, latest_reference_odom_msg_, t);
  }

  bool calculate_slip(
    const tf2::Transform & transform1_wheelodom,
    const tf2::Transform & transform1_ref,
    const tf2::Transform & transform2_wheelodom,
    const tf2::Transform & transform2_ref,
    float & slip) const
  {
    tf2::Transform displacement_wheelodom = transform1_wheelodom.inverse() * transform2_wheelodom;
    tf2::Transform displacement_ref = transform1_ref.inverse() * transform2_ref;

    double dsl_wheelodom = 0.0;
    double dsr_wheelodom = 0.0;
    double dsl_ref = 0.0;
    double dsr_ref = 0.0;
    compute_traveled_wheeldistances(displacement_wheelodom, dsl_wheelodom, dsr_wheelodom);
    compute_traveled_wheeldistances(displacement_ref, dsl_ref, dsr_ref);

    if (std::fabs(dsl_wheelodom) < sample_distance_ &&
      std::fabs(dsr_wheelodom) < sample_distance_ &&
      std::fabs(dsl_ref) < sample_distance_ &&
      std::fabs(dsr_ref) < sample_distance_)
    {
      return false;
    }

    const double right_slip = compute_normalized_slip_component(dsr_wheelodom, dsr_ref);
    const double left_slip = compute_normalized_slip_component(dsl_wheelodom, dsl_ref);

    // Average
    // slip = static_cast<float>((right_slip + left_slip) / 2.0);

    // Norm 2
    slip = std::sqrt(right_slip * right_slip + left_slip * left_slip) / std::sqrt(2.0);

    // Max
    // slip = static_cast<float>(std::max(right_slip, left_slip));
    return true;
  }

  double compute_normalized_slip_component(double wheelodom_distance, double ref_distance) const {
    const double scale = std::max(std::fabs(wheelodom_distance), std::fabs(ref_distance));
    // if (scale <= std::numeric_limits<double>::epsilon()) {
    if (scale <= min_distance_threshold_) {
      return 0.0;
    }

    return std::fabs(wheelodom_distance - ref_distance) / scale;
  }

  void compute_traveled_wheeldistances(
    const tf2::Transform & displacement,
    double & dsl,
    double & dsr) const
  {
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Matrix3x3 rot_mat(displacement.getRotation());
    rot_mat.getRPY(roll, pitch, yaw);

    tf2::Vector3 dv = displacement.getOrigin();
    dv.setZ(0.0);
    const double ds = dv.length();

    dsr = ds + yaw * wheel_separation_ / 2.0;
    dsl = ds - yaw * wheel_separation_ / 2.0;
  }

  bool check_minimum_velocity() {
    if (wheel_odom_config_.configured_type == OdomSourceType::Frame
      || reference_odom_config_.configured_type == OdomSourceType::Frame) {
      // If using TF frames, we don't have direct access to velocity information.
      // We could compute it from the change in position over time, but that would be more complex and less accurate.
      // For now, we will skip the minimum velocity check when using TF frames.
      return true;
    }

    if (!latest_wheel_odom_msg_) {
      return false;
    }

    const double vx = latest_wheel_odom_msg_->twist.twist.linear.x;
    const double vy = latest_wheel_odom_msg_->twist.twist.linear.y;
    const double vz = latest_wheel_odom_msg_->twist.twist.linear.z;
    const double velocity = std::sqrt(vx * vx + vy * vy + vz * vz);

    if (velocity < min_velocity_threshold_) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Wheel odometry velocity %.4f m/s below minimum threshold %.4f m/s. Skipping slip calculation.",
        velocity, min_velocity_threshold_);
      return false;
    }

    return true;
  }

  bool initialized_{false};
  bool skip_iteration_{false};
  double wheel_separation_;
  double sample_distance_;
  double min_distance_threshold_;
  double min_acceleration_threshold_;
  double odom_timeout_sec_;
  double min_velocity_threshold_;
  double tf_poll_period_sec_;
  std::string robot_frame_;
  std::string wheel_odom_;
  std::string reference_odom_;
  std::string output_topic_;
  std::string feature_service_name_;
  OdomSourceConfig wheel_odom_config_;
  OdomSourceConfig reference_odom_config_;

  tf2::Transform transform_last_wheelodom_;
  tf2::Transform transform_last_ref_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<soislip_interfaces::msg::SOINNSample>::SharedPtr sample_pub_;
  rclcpp::Client<soislip_interfaces::srv::GetCellFeatures>::SharedPtr feature_client_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr reference_odom_sub_;
  nav_msgs::msg::Odometry::SharedPtr latest_wheel_odom_msg_;
  nav_msgs::msg::Odometry::SharedPtr latest_reference_odom_msg_;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotExperienceCollectorNode>());
  rclcpp::shutdown();
  return 0;
}
