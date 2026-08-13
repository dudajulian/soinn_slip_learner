#include <memory>
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "grid_map_core/grid_map_core.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "rclcpp/rclcpp.hpp"

#include "soislip_interfaces/srv/get_map_features.hpp"
#include "soislip_interfaces/srv/predict_batch.hpp"

class SlipPredictionManagerNode : public rclcpp::Node {
public:
  SlipPredictionManagerNode()
  : Node("slip_prediction_manager") {
    this->declare_parameter("reference_map_topic", "/soislip/feature_map");
    this->declare_parameter("slip_layer_name", "slip_prediction");
    this->declare_parameter("confidence_layer_name", "slip_confidence");
    this->declare_parameter("map_feature_service_name", "get_map_features");
    this->declare_parameter("predict_batch_service_name", "predict_batch");
    this->declare_parameter("prediction_period_sec", 5.0);

    this->get_parameter("reference_map_topic", ref_map_topic_);
    this->get_parameter("slip_layer_name", slip_layer_name_);
    this->get_parameter("confidence_layer_name", confidence_layer_name_);
    this->get_parameter("map_feature_service_name", map_feature_service_name_);
    this->get_parameter("predict_batch_service_name", predict_batch_service_name_);
    this->get_parameter("prediction_period_sec", prediction_period_sec_);

    slip_map_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
      "/soislip/prediction_map", 10);

    map_sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
      ref_map_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&SlipPredictionManagerNode::handle_map_update, this, std::placeholders::_1));

    map_feature_client_ = this->create_client<soislip_interfaces::srv::GetMapFeatures>(
      map_feature_service_name_);
    predict_batch_client_ = this->create_client<soislip_interfaces::srv::PredictBatch>(
      predict_batch_service_name_);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(prediction_period_sec_),
      std::bind(&SlipPredictionManagerNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "slip_prediction_manager started");
  }

private:
  void handle_map_update(const grid_map_msgs::msg::GridMap::SharedPtr msg) {
    grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(*msg, map);
    std::scoped_lock<std::mutex> lock(map_mutex_);
    latest_map_ = map;
    has_map_ = true;
  }

  void timer_callback() {
    if (request_in_flight_) {
      return;
    }

    if (!map_feature_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Service '%s' is not available", map_feature_service_name_.c_str());
      return;
    }
    if (!predict_batch_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Service '%s' is not available", predict_batch_service_name_.c_str());
      return;
    }
    double rotation = 0.0F; //TODO: Get rotation from robot pose.


    request_in_flight_ = true;
    auto map_request = std::make_shared<soislip_interfaces::srv::GetMapFeatures::Request>();
    map_request->rotation = rotation; 
    map_feature_client_->async_send_request(
      map_request,
      std::bind(&SlipPredictionManagerNode::handle_map_features_response, this, std::placeholders::_1));
  }

  void handle_map_features_response(
    rclcpp::Client<soislip_interfaces::srv::GetMapFeatures>::SharedFuture future)
  {
    try {
      auto map_response = future.get();
      if (!map_response->success.data) {
        RCLCPP_WARN(
          this->get_logger(), "get_map_features failed: %s", map_response->message.data.c_str());
        request_in_flight_ = false;
        return;
      }

      if (map_response->feature_dim <= 0 || map_response->features.data.empty() ||
        map_response->positions.empty())
      {
        RCLCPP_WARN(this->get_logger(), "get_map_features returned empty payload");
        request_in_flight_ = false;
        return;
      }

      const size_t feature_count = map_response->features.data.size();
      if (feature_count % static_cast<size_t>(map_response->feature_dim) != 0U) {
        RCLCPP_WARN(this->get_logger(), "Invalid flattened feature payload size=%zu dim=%d",
          feature_count, map_response->feature_dim);
        request_in_flight_ = false;
        return;
      }

      auto predict_request = std::make_shared<soislip_interfaces::srv::PredictBatch::Request>();
      predict_request->feature_dim = map_response->feature_dim;
      predict_request->features.data = map_response->features.data;

      auto positions = std::make_shared<std::vector<geometry_msgs::msg::Point>>(map_response->positions);
      predict_batch_client_->async_send_request(
        predict_request,
        [this, positions](rclcpp::Client<soislip_interfaces::srv::PredictBatch>::SharedFuture predict_future) {
          this->handle_predict_batch_response(std::move(positions), predict_future);
        });
    } catch (const std::exception & ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to process get_map_features response: %s", ex.what());
      request_in_flight_ = false;
    }
  }

  void handle_predict_batch_response(
    const std::shared_ptr<std::vector<geometry_msgs::msg::Point>> & positions,
    rclcpp::Client<soislip_interfaces::srv::PredictBatch>::SharedFuture future)
  {
    try {
      auto response = future.get();
      if (!response->success.data) {
        RCLCPP_WARN(this->get_logger(), "predict_batch failed: %s", response->message.data.c_str());
        request_in_flight_ = false;
        return;
      }

      const size_t position_count = positions->size();
      const size_t prediction_count = response->predictions.data.size();
      const size_t confidence_count = response->confidence_scores.data.size();

      if (position_count == 0U || prediction_count == 0U || confidence_count == 0U) {
        request_in_flight_ = false;
        return;
      }

      if (position_count != prediction_count || prediction_count != confidence_count) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Size mismatch: positions=%zu predictions=%zu confidence_scores=%zu",
          position_count, prediction_count, confidence_count);
        request_in_flight_ = false;
        return;
      }

      const size_t n = position_count;

      publish_slip_prediction_map(
        *positions, response->predictions.data, response->confidence_scores.data, n);
    } catch (const std::exception & ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to process predict_batch response: %s", ex.what());
    }

    request_in_flight_ = false;
  }

  void publish_slip_prediction_map(
    const std::vector<geometry_msgs::msg::Point> & positions,
    const std::vector<float> & predictions,
    const std::vector<float> & confidence_scores,
    size_t n)
  {
    grid_map::GridMap map;
    {
      std::scoped_lock<std::mutex> lock(map_mutex_);
      if (!has_map_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "No elevation map available to publish prediction layer");
        return;
      }
      map = latest_map_;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (!map.exists(slip_layer_name_)) {
      map.add(slip_layer_name_, nan);
    } else {
      map[slip_layer_name_].setConstant(nan);
    }
    if (!map.exists(confidence_layer_name_)) {
      map.add(confidence_layer_name_, nan);
    } else {
      map[confidence_layer_name_].setConstant(nan);
    }

    for (size_t i = 0; i < n; ++i) {
      const grid_map::Position position(
        static_cast<double>(positions[i].x),
        static_cast<double>(positions[i].y));
      grid_map::Index index;
      if (!map.getIndex(position, index)) {
        continue;
      }
      map.at(slip_layer_name_, index) = predictions[i];
      map.at(confidence_layer_name_, index) = confidence_scores[i];
    }

    auto msg = grid_map::GridMapRosConverter::toMessage(map);
    if (!msg) {
      return;
    }
    slip_map_pub_->publish(*msg);
  }

  bool request_in_flight_{false};
  double prediction_period_sec_{1.0};
  std::string slip_prediction_map_topic_;
  std::string ref_map_topic_;
  std::string slip_layer_name_;
  std::string confidence_layer_name_;
  std::string map_feature_service_name_;
  std::string predict_batch_service_name_;

  bool has_map_{false};
  grid_map::GridMap latest_map_;
  std::mutex map_mutex_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr slip_map_pub_;
  rclcpp::Client<soislip_interfaces::srv::GetMapFeatures>::SharedPtr map_feature_client_;
  rclcpp::Client<soislip_interfaces::srv::PredictBatch>::SharedPtr predict_batch_client_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SlipPredictionManagerNode>());
  rclcpp::shutdown();
  return 0;
}
