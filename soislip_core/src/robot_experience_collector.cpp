#include <memory>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "soislip_interfaces/srv/get_cell_features.hpp"

class RobotExperienceCollectorNode : public rclcpp::Node {
public:
  RobotExperienceCollectorNode()
  : Node("robot_experience_collector_node") {
    this->declare_parameter("wheel_separation", 0.3);
    this->declare_parameter("robot_frame", "base_link");
    this->declare_parameter("wheelodom_frame", "odom");
    this->declare_parameter("reference_frame", "map");
    this->declare_parameter("movement_threshold", 0.05);
    this->declare_parameter("sample_topic", "/experience_samples");
    this->declare_parameter("feature_service_name", "get_cell_features");
    this->declare_parameter("collector_period_sec", 0.1);

    this->get_parameter("wheel_separation", wheel_separation_);
    this->get_parameter("robot_frame", robot_frame_);
    this->get_parameter("wheelodom_frame", wheelodom_frame_);
    this->get_parameter("reference_frame", reference_frame_);
    this->get_parameter("movement_threshold", movement_threshold_);
    this->get_parameter("sample_topic", sample_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("collector_period_sec", collector_period_sec_);

    sample_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(sample_topic_, 10);
    feature_client_ = this->create_client<soislip_interfaces::srv::GetCellFeatures>(feature_service_name_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(collector_period_sec_),
      std::bind(&RobotExperienceCollectorNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "robot_experience_collector_node started");
  }

private:
  void timer_callback() {
    tf2::Transform current_wheelodom;
    tf2::Transform current_ref;

    if (!get_robot_transforms(current_wheelodom, current_ref, robot_frame_, wheelodom_frame_, reference_frame_)) {
      skip_iteration_ = true;
      return;
    }

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

    float slip = 0.0F;
    if (calculate_slip(transform_last_wheelodom_, transform_last_ref_, current_wheelodom, current_ref, slip)) {
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

              std_msgs::msg::Float32MultiArray sample;
              sample.data.reserve(response->features.data.size() + 1);
              sample.data.push_back(slip);
              sample.data.insert(sample.data.end(), response->features.data.begin(), response->features.data.end());
              sample_pub_->publish(sample);
            } catch (const std::exception & ex) {
              RCLCPP_WARN(this->get_logger(), "Feature service response failed: %s", ex.what());
            }
          });
      }
    }

    transform_last_wheelodom_ = current_wheelodom;
    transform_last_ref_ = current_ref;
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

  bool get_robot_transforms(
    tf2::Transform & transform_wheelodom,
    tf2::Transform & transform_ref,
    const std::string & robot_frame,
    const std::string & wheelodom_frame,
    const std::string & ref_frame,
    const tf2::TimePoint & t = tf2::TimePointZero)
  {
    try {
      tf2::fromMsg(tf_buffer_->lookupTransform(robot_frame, wheelodom_frame, t).transform, transform_wheelodom);
      tf2::fromMsg(tf_buffer_->lookupTransform(robot_frame, ref_frame, t).transform, transform_ref);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Could not get transforms: %s", ex.what());
      return false;
    }
    return true;
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

    if (std::fabs(dsl_wheelodom) < movement_threshold_ &&
      std::fabs(dsr_wheelodom) < movement_threshold_ &&
      std::fabs(dsl_ref) < movement_threshold_ &&
      std::fabs(dsr_ref) < movement_threshold_)
    {
      return false;
    }

    const double right_scale = std::max(std::fabs(dsr_wheelodom), std::fabs(dsr_ref));
    const double left_scale = std::max(std::fabs(dsl_wheelodom), std::fabs(dsl_ref));
    if (right_scale <= 0.0 || left_scale <= 0.0) {
      return false;
    }

    const double right_slip = std::fabs(dsr_wheelodom - dsr_ref) / right_scale;
    const double left_slip = std::fabs(dsl_wheelodom - dsl_ref) / left_scale;

    slip = static_cast<float>((right_slip + left_slip) / 2.0);
    return true;
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

  bool initialized_{false};
  bool skip_iteration_{false};
  double wheel_separation_{1.0};
  double movement_threshold_{0.05};
  double collector_period_sec_{0.1};
  std::string robot_frame_;
  std::string wheelodom_frame_;
  std::string reference_frame_;
  std::string sample_topic_;
  std::string feature_service_name_;

  tf2::Transform transform_last_wheelodom_;
  tf2::Transform transform_last_ref_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr sample_pub_;
  rclcpp::Client<soislip_interfaces::srv::GetCellFeatures>::SharedPtr feature_client_;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotExperienceCollectorNode>());
  rclcpp::shutdown();
  return 0;
}
