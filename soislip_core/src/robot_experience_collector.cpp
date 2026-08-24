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
  
OdomSourceType parse_odom_source_type(const std::string & value) {
  if (value == "frame") {
    return OdomSourceType::Frame;
  }
  if (value == "topic") {
    return OdomSourceType::Topic;
  }
  throw std::invalid_argument("Unknown odometry source type: " + value + ". Expected 'frame' or 'topic'.");
}

class RobotExperienceCollectorNode : public rclcpp::Node {
public:
  RobotExperienceCollectorNode()
  : Node("robot_experience_collector_node") {
    this->declare_parameter("wheel_separation", 0.3);
    this->declare_parameter("wheel_x_offset", 0.0);
    this->declare_parameter("robot_frame", "base_link");
    this->declare_parameter("wheel_odom", "odom");
    this->declare_parameter("reference_odom", "map");
    this->declare_parameter("wheel_odom_source", "frame");
    this->declare_parameter("reference_odom_source", "frame");
    this->declare_parameter("sample_distance", 0.1);
    this->declare_parameter("min_distance_threshold", 0.05);
    this->declare_parameter("min_velocity_threshold", 0.05);
    this->declare_parameter("max_velocity_threshold", 2.0);
    this->declare_parameter("odom_timeout_sec", 0.1);
    this->declare_parameter("output_topic", "/experience_samples");
    this->declare_parameter("feature_service_name", "get_cell_features");
    this->declare_parameter("tf_poll_period_sec", 0.05); // 20 Hz like the /tf topic

    std::string wheel_odom_source;
    std::string reference_odom_source;
    this->get_parameter("wheel_separation", wheel_separation_);
    this->get_parameter("wheel_x_offset", wheel_x_offset_);
    this->get_parameter("robot_frame", robot_frame_);
    this->get_parameter("wheel_odom", wheel_odom_);
    this->get_parameter("reference_odom", reference_odom_);
    this->get_parameter("wheel_odom_source", wheel_odom_source);
    this->get_parameter("reference_odom_source", reference_odom_source);
    this->get_parameter("sample_distance", sample_distance_);
    this->get_parameter("min_distance_threshold", min_distance_threshold_);
    this->get_parameter("odom_timeout_sec", odom_timeout_sec_);
    this->get_parameter("min_velocity_threshold", min_wheel_velocity_threshold_);
    this->get_parameter("max_velocity_threshold", max_ref_velocity_threshold_);
    this->get_parameter("output_topic", output_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("tf_poll_period_sec", tf_poll_period_sec_);

    wheel_odom_config_.name = wheel_odom_;
    wheel_odom_config_.configured_type = parse_odom_source_type(wheel_odom_source);
    reference_odom_config_.name = reference_odom_;
    reference_odom_config_.configured_type = parse_odom_source_type(reference_odom_source);


    sample_pub_ = this->create_publisher<soislip_interfaces::msg::SOINNSample>(output_topic_, 10);
    feature_client_ = this->create_client<soislip_interfaces::srv::GetCellFeatures>(feature_service_name_);
    
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    initialize_odometry_subscribers();
  
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(tf_poll_period_sec_),
      std::bind(&RobotExperienceCollectorNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "robot_experience_collector_node started");
  }

private:
  void timer_callback() {
    tf2::Transform new_tf_wheelodom;
    tf2::Transform new_tf_ref;
    double new_timestamp;
    double dt_wheel2ref;
    
    // If the feature service is not ready, log a warning and skip this iteration.
    if (!feature_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000, "Feature service '%s' not available",
        feature_service_name_.c_str());
      skip_iteration_ = true;
      return;
    } 

    // Get the current transforms for wheel odometry and reference odometry.
    if (!get_robot_transforms(new_tf_wheelodom, new_tf_ref, new_timestamp, dt_wheel2ref)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Failed to get robot transforms (wheel_odom='%s', reference_odom='%s')",
        wheel_odom_.c_str(), reference_odom_.c_str());
      return;
    }

    // Log the current transforms for debugging purposes.
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Got robot transforms in %.2f sec delta: wheel_odom=(%.2f, %.2f, %.2f), reference_odom=(%.2f, %.2f, %.2f)",
      dt_wheel2ref,
      new_tf_wheelodom.getOrigin().x(), new_tf_wheelodom.getOrigin().y(), new_tf_wheelodom.getOrigin().z(),
      new_tf_ref.getOrigin().x(), new_tf_ref.getOrigin().y(), new_tf_ref.getOrigin().z());
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "skip_iteration_=%s", skip_iteration_ ? "true" : "false");

    // If this is the first iteration after a reset, we skip processing to avoid publishing a sample with zero displacement.
    if (skip_iteration_) {
      skip_iteration_ = false;
      last_tf_wheelodom_ = new_tf_wheelodom;
      last_tf_ref_ = new_tf_ref;
      last_timestamp_ = new_timestamp;
      return;
    }
    // Check if the robot's velocity is above the minimum threshold before proceeding with slip calculation.
    double dt = std::chrono::duration<double>(new_timestamp - last_timestamp_).count();
    if (!check_minimum_velocity(last_tf_wheelodom_, new_tf_wheelodom, dt)) {
      skip_iteration_ = true;
      return;
    }
    if (!check_maximum_velocity(last_tf_ref_, new_tf_ref, dt)) {
      skip_iteration_ = true;
      return;
    }
    // Calculate the slip based on the last and current transforms.
    float rslip, lslip;
    if (!calculate_slip(last_tf_wheelodom_, last_tf_ref_, new_tf_wheelodom, new_tf_ref, rslip, lslip)) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Waiting for sufficient movement (successive transforms) to calculate slip.");
      return;
    }
    std::array<float, 2> wheel_slips = {lslip, rslip};
    
    double robot_yaw = get_yaw_from_transform(last_tf_ref_.inverse());
    std::array<geometry_msgs::msg::Point, 2> wheel_positions = calculate_wheel_position(last_tf_ref_, wheel_x_offset_);
      for (size_t i = 0; i < 2; ++i) {
        float slip = wheel_slips[i];
        geometry_msgs::msg::Point sample_pos = wheel_positions[i];
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Requesting features for wheel %zu at position (%.2f, %.2f, %.2f) with slip %.3f", 
          i, sample_pos.x, sample_pos.y, sample_pos.z, slip);
        auto request = std::make_shared<soislip_interfaces::srv::GetCellFeatures::Request>();
        request->position = sample_pos;
        request->rotation = static_cast<double>(robot_yaw);

        // Asynchronously send the feature request and handle the response in a callback.
        feature_client_->async_send_request(
          request,
          [this, slip, request](rclcpp::Client<soislip_interfaces::srv::GetCellFeatures>::SharedFuture future) 
          {
            soislip_interfaces::msg::SOINNSample sample;
            // Try to publish the experience sample with the features and slip label.
            try {
              auto response = future.get();
              if (!response->success.data) {
                throw std::runtime_error(response->message.data.c_str());
              }
              sample.features.assign(response->features.data.begin(), response->features.data.end());
              sample.label = slip;
              sample.has_label = true;
              sample.position.point = request->position;
              sample.has_position = true;
              sample.position.header.stamp = this->now();
              sample.position.header.frame_id = "map"; // TODO: Make this configurable if needed;
              sample_pub_->publish(sample);
              RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
              "Publishing experience sample with %zu features and label %.3f", sample.features.size(), sample.label);
            }
            // Catch any exceptions that occur while processing the response and log a warning.
            catch (const std::exception & ex) {
              RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
              "Feature service response failed: %s", ex.what());
            }
          }
        );
      }
    // Update the last transforms for the next iteration.
    last_tf_wheelodom_ = new_tf_wheelodom;
    last_tf_ref_ = new_tf_ref;
    last_timestamp_ = new_timestamp;
  }
  

  // ----------------------------------------------------------------------------------------------
  // Odometry Subscribers and Callbacks
  // ----------------------------------------------------------------------------------------------

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
    

  // ----------------------------------------------------------------------------------------------
  // Transform Retrieval
  // ----------------------------------------------------------------------------------------------

  bool get_robot_transforms(
    tf2::Transform & transform_wheelodom,
    tf2::Transform & transform_ref,
    double & t,
    double & dt_wheel2ref)
  {
    double t1, t2;
    rclcpp::Time lookup_time = this->now();
    bool sucess = get_single_transform(transform_wheelodom, t1, wheel_odom_config_, latest_wheel_odom_msg_, lookup_time) &&
      get_single_transform(transform_ref, t2, reference_odom_config_, latest_reference_odom_msg_, lookup_time);
    t = (t1 + t2) / 2.0;
    dt_wheel2ref = t2 - t1;
    return sucess;
  }

  bool get_single_transform(
    tf2::Transform & transform,
    double & t,
    const OdomSourceConfig & config,
    const nav_msgs::msg::Odometry::SharedPtr & odom_msg,
    const rclcpp::Time & lookup_time)
  {
    if (config.configured_type == OdomSourceType::Topic) {
      return get_transform_from_topic(transform, t, odom_msg, config.name, lookup_time);
    }
    return get_transform_from_frame(transform, t, config.name, lookup_time);
  }

  // Get the transform from the odometry topic.
  bool get_transform_from_topic(
    tf2::Transform & transform,
    double & t,
    const nav_msgs::msg::Odometry::SharedPtr & odom_msg,
    const std::string & source_name,
    const rclcpp::Time & lookup_time)
  {
    if (!odom_msg) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Waiting for Odometry messages on '%s'",
        source_name.c_str());
      return false;
    }

    const rclcpp::Time msg_stamp(odom_msg->header.stamp);
    const rclcpp::Duration timeout = rclcpp::Duration::from_seconds(odom_timeout_sec_);
    if ((lookup_time - msg_stamp) > timeout) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Odometry topic '%s' timed out. Latest message age exceeds %.3f s.",
        source_name.c_str(), odom_timeout_sec_);
      return false;
    }

    if (!odom_msg->child_frame_id.empty() || odom_msg->child_frame_id != robot_frame_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Odometry topic '%s' has missing or mismatched child_frame_id. Expected '%s'.",
        source_name.c_str(), robot_frame_.c_str());
      return false;
    }

    tf2::fromMsg(odom_msg->pose.pose, transform);
    t = msg_stamp.seconds();
    return true;
  }

  // Get the transform from the TF2 buffer.
  bool get_transform_from_frame(
    tf2::Transform & transform,
    double & t,
    const std::string & source_frame,
    const rclcpp::Time & lookup_time)
  {
    (void) lookup_time; // Unused parameter, but kept for potential future use or consistency with other methods.
    geometry_msgs::msg::TransformStamped stamped_tf;
    try {
      // stamped_tf = tf_buffer_->lookupTransform(robot_frame_, source_frame, lookup_time, rclcpp::Duration::from_seconds(odom_timeout_sec_));
      stamped_tf = tf_buffer_->lookupTransform(robot_frame_, source_frame, tf2::TimePointZero);
      tf2::fromMsg(stamped_tf.transform, transform);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Could not get transform from frame '%s': %s",
        source_frame.c_str(), ex.what());
      return false;
    }
    t = stamped_tf.header.stamp.sec + stamped_tf.header.stamp.nanosec * 1e-9;
    return true;
  }


  // ----------------------------------------------------------------------------------------------
  // Slip Calculation from Transforms
  // ----------------------------------------------------------------------------------------------

  bool calculate_slip(
    const tf2::Transform & transform1_wheelodom,
    const tf2::Transform & transform1_ref,
    const tf2::Transform & transform2_wheelodom,
    const tf2::Transform & transform2_ref,
    float & rslip,
    float & lslip) 
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
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Slip calculation failed: Displacement too small (wheelodom: left=%.4f, right=%.4f; ref: left=%.4f, right=%.4f)",
        dsl_wheelodom, dsr_wheelodom, dsl_ref, dsr_ref);
      return false;
    }

    const double right_slip = compute_normalized_slip_component(dsr_wheelodom, dsr_ref);
    const double left_slip = compute_normalized_slip_component(dsl_wheelodom, dsl_ref);

    // Average
    // slip = static_cast<float>((right_slip + left_slip) / 2.0);

    // Norm 2
    rslip = static_cast<float>(right_slip);
    lslip = static_cast<float>(left_slip);

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
    double yaw = get_yaw_from_transform(displacement);

    tf2::Vector3 dv = displacement.getOrigin();
    dv.setZ(0.0);
    const double ds = dv.length();

    dsr = ds + yaw * wheel_separation_ / 2.0;
    dsl = ds - yaw * wheel_separation_ / 2.0;
  }

  // ----------------------------------------------------------------------------------------------
  // Helper Functions
  // ----------------------------------------------------------------------------------------------

  static double get_yaw_from_transform(const tf2::Transform & transform) {
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Matrix3x3 rot_mat(transform.getRotation());
    rot_mat.getRPY(roll, pitch, yaw);
    return yaw;
  }
    
  static geometry_msgs::msg::Point calculate_midpoint_position(
    const tf2::Transform & transform1_ref,
    const tf2::Transform & transform2_ref)
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

  std::array<geometry_msgs::msg::Point, 2> calculate_wheel_position(
    const tf2::Transform & transform1_ref, double x_offset = 0.0, double z_offset = 0.0)
    {
      tf2::Vector3 sample_pos = transform1_ref.inverse().getOrigin();
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Calculate features from position (%.2f, %.2f, %.2f)", 
        sample_pos.x(), sample_pos.y(), sample_pos.z());
      const tf2::Vector3 left_offset(x_offset, wheel_separation_ / 2.0, z_offset);
      const tf2::Vector3 right_offset(x_offset, -wheel_separation_ / 2.0, z_offset);
      const tf2::Vector3 point_left = transform1_ref.inverse() * left_offset;
      const tf2::Vector3 point_right = transform1_ref.inverse() * right_offset;

      geometry_msgs::msg::Point point_left_msg;
      point_left_msg.x = point_left.x();
      point_left_msg.y = point_left.y();
      point_left_msg.z = point_left.z();

      geometry_msgs::msg::Point point_right_msg;
      point_right_msg.x = point_right.x();
      point_right_msg.y = point_right.y();
      point_right_msg.z = point_right.z();

      return {point_left_msg, point_right_msg};
  }

  bool check_minimum_velocity(const tf2::Transform & t1, const tf2::Transform & t2, const double dt) {
    float ds = t1.inverse().getOrigin().distance(t2.inverse().getOrigin());
    float velocity = ds / dt;
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Wheel odometry displacement %.4f m over %.4f s, velocity %.4f m/s", ds, dt, velocity);
    if (velocity < min_wheel_velocity_threshold_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Wheel odometry velocity %.4f m/s below minimum threshold %.4f m/s. Skipping slip calculation.", 
        velocity, min_wheel_velocity_threshold_);
      return false;
    }
    return true;
  }

  bool check_maximum_velocity(const tf2::Transform & t1, const tf2::Transform & t2, const double dt) {
    float ds = t1.inverse().getOrigin().distance(t2.inverse().getOrigin());
    float velocity = ds / dt;
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Reference odometry displacement %.4f m over %.4f s, velocity %.4f m/s", ds, dt, velocity);
    if (velocity > max_ref_velocity_threshold_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Reference odometry velocity %.4f m/s above maximum threshold %.4f m/s. Skipping slip calculation.", 
        velocity, max_ref_velocity_threshold_);
      return false;
    }
    return true;
  }

  bool skip_iteration_{true};
  double wheel_separation_; // Distance between the left and right wheels of the robot (default: 0.3m)
  double sample_distance_; // Minimum distance traveled before publishing a new experience sample (default: 0.1m)
  double min_distance_threshold_; // Minimum distance traveled per wheel in any odometry source to avoid division by close to zero (default: 0.05m)
  double odom_timeout_sec_;
  double min_wheel_velocity_threshold_; // Minimum wheel odometry velocity to consider for slip calculation (default: 0.05 m/s)
  double max_ref_velocity_threshold_; // Maximum reference odometry velocity to consider for slip calculation (default: 2.0 m/s)
  double tf_poll_period_sec_; // How often to poll for transforms in seconds (default: 0.05s = 20Hz)
  std::string robot_frame_;
  std::string wheel_odom_;
  std::string reference_odom_;
  std::string output_topic_;
  std::string feature_service_name_;
  OdomSourceConfig wheel_odom_config_;
  OdomSourceConfig reference_odom_config_;
  double wheel_x_offset_;

  // Transforms from odometry source frames to the robot frame, used to compute displacements and slips.
  tf2::Transform last_tf_wheelodom_;
  tf2::Transform last_tf_ref_;
  double last_timestamp_;

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
