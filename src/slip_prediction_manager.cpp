#include <memory>
#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "soinn_slip_learner/srv/get_map_features.hpp"
#include "soinn_slip_learner/srv/predict_batch.hpp"

class SlipPredictionManagerNode : public rclcpp::Node {
public:
  SlipPredictionManagerNode()
  : Node("slip_prediction_manager") {
    this->declare_parameter("output_topic", "/gridmap_with_predictions");
    this->declare_parameter("map_feature_service_name", "get_map_features");
    this->declare_parameter("predict_batch_service_name", "predict_batch");
    this->declare_parameter("prediction_period_sec", 1.0);

    this->get_parameter("output_topic", output_topic_);
    this->get_parameter("map_feature_service_name", map_feature_service_name_);
    this->get_parameter("predict_batch_service_name", predict_batch_service_name_);
    this->get_parameter("prediction_period_sec", prediction_period_sec_);

    prediction_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(output_topic_, 10);

    map_feature_client_ = this->create_client<soinn_slip_learner::srv::GetMapFeatures>(
      map_feature_service_name_);
    predict_batch_client_ = this->create_client<soinn_slip_learner::srv::PredictBatch>(
      predict_batch_service_name_);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(prediction_period_sec_),
      std::bind(&SlipPredictionManagerNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "slip_prediction_manager started");
  }

private:
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

    request_in_flight_ = true;
    auto map_request = std::make_shared<soinn_slip_learner::srv::GetMapFeatures::Request>();
    map_feature_client_->async_send_request(
      map_request,
      std::bind(&SlipPredictionManagerNode::handle_map_features_response, this, std::placeholders::_1));
  }

  void handle_map_features_response(
    rclcpp::Client<soinn_slip_learner::srv::GetMapFeatures>::SharedFuture future)
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

      auto predict_request = std::make_shared<soinn_slip_learner::srv::PredictBatch::Request>();
      predict_request->feature_dim = map_response->feature_dim;
      predict_request->features.data = map_response->features.data;

      auto positions = std::make_shared<std::vector<geometry_msgs::msg::Point>>(map_response->positions);
      predict_batch_client_->async_send_request(
        predict_request,
        [this, positions](rclcpp::Client<soinn_slip_learner::srv::PredictBatch>::SharedFuture predict_future) {
          this->handle_predict_batch_response(std::move(positions), predict_future);
        });
    } catch (const std::exception & ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to process get_map_features response: %s", ex.what());
      request_in_flight_ = false;
    }
  }

  void handle_predict_batch_response(
    const std::shared_ptr<std::vector<geometry_msgs::msg::Point>> & positions,
    rclcpp::Client<soinn_slip_learner::srv::PredictBatch>::SharedFuture future)
  {
    try {
      auto response = future.get();
      if (!response->success.data) {
        RCLCPP_WARN(this->get_logger(), "predict_batch failed: %s", response->message.data.c_str());
        request_in_flight_ = false;
        return;
      }

      const size_t n = std::min(positions->size(), response->predictions.data.size());
      if (n == 0U) {
        request_in_flight_ = false;
        return;
      }

      std_msgs::msg::Float32MultiArray out;
      out.data.reserve(n * 3U);
      for (size_t i = 0; i < n; ++i) {
        out.data.push_back(static_cast<float>((*positions)[i].x));
        out.data.push_back(static_cast<float>((*positions)[i].y));
        out.data.push_back(response->predictions.data[i]);
      }

      prediction_pub_->publish(out);
    } catch (const std::exception & ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to process predict_batch response: %s", ex.what());
    }

    request_in_flight_ = false;
  }

  bool request_in_flight_{false};
  double prediction_period_sec_{1.0};
  std::string output_topic_;
  std::string map_feature_service_name_;
  std::string predict_batch_service_name_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr prediction_pub_;
  rclcpp::Client<soinn_slip_learner::srv::GetMapFeatures>::SharedPtr map_feature_client_;
  rclcpp::Client<soinn_slip_learner::srv::PredictBatch>::SharedPtr predict_batch_client_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SlipPredictionManagerNode>());
  rclcpp::shutdown();
  return 0;
}
