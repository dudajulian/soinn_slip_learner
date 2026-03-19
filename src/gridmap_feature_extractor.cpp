#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

#include <rclcpp/rclcpp.hpp>

#include "soinn_slip_learner/srv/get_cell_features.hpp"
#include "soinn_slip_learner/srv/get_map_features.hpp"

class GridmapFeatureExtractorNode : public rclcpp::Node {
public:
  GridmapFeatureExtractorNode()
  : Node("gridmap_feature_extractor_node") {
    this->declare_parameter("elevation_map_topic", "/elevation_map");
    this->declare_parameter("feature_service_name", "get_cell_features");
    this->declare_parameter("map_feature_service_name", "get_map_features");
    this->declare_parameter("feature_radius", 0.5);

    this->get_parameter("elevation_map_topic", elevation_map_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("map_feature_service_name", map_feature_service_name_);
    this->get_parameter("feature_radius", feature_radius_);

    sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
      elevation_map_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridmapFeatureExtractorNode::callback, this, std::placeholders::_1));

    service_ = this->create_service<soinn_slip_learner::srv::GetCellFeatures>(
      feature_service_name_,
      std::bind(
        &GridmapFeatureExtractorNode::handle_get_cell_features,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    map_service_ = this->create_service<soinn_slip_learner::srv::GetMapFeatures>(
      map_feature_service_name_,
      std::bind(
        &GridmapFeatureExtractorNode::handle_get_map_features,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "gridmap_feature_extractor_node started");
  }

private:
  void callback(const grid_map_msgs::msg::GridMap::SharedPtr msg) {
    grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(*msg, map);
    std::scoped_lock<std::mutex> lock(map_mutex_);
    latest_map_ = map;
    has_map_ = true;
  }

  void handle_get_cell_features(
    const std::shared_ptr<soinn_slip_learner::srv::GetCellFeatures::Request> request,
    std::shared_ptr<soinn_slip_learner::srv::GetCellFeatures::Response> response)
  {
    grid_map::GridMap map;
    {
      std::scoped_lock<std::mutex> lock(map_mutex_);
      if (!has_map_) {
        response->success.data = false;
        response->message.data = "No grid map received yet";
        return;
      }
      map = latest_map_;
    }

    const grid_map::Position center(
      static_cast<double>(request->position.x),
      static_cast<double>(request->position.y));

    std::vector<float> features = extract_features(map, center);
    if (features.empty()) {
      response->success.data = false;
      response->message.data = "Could not extract features at requested position";
      return;
    }

    response->features.data = features;
    response->success.data = true;
    response->message.data = "ok";
  }

  void handle_get_map_features(
    const std::shared_ptr<soinn_slip_learner::srv::GetMapFeatures::Request> request,
    std::shared_ptr<soinn_slip_learner::srv::GetMapFeatures::Response> response)
  {
    (void)request;
    grid_map::GridMap map;
    {
      std::scoped_lock<std::mutex> lock(map_mutex_);
      if (!has_map_) {
        response->success.data = false;
        response->message.data = "No grid map received yet";
        response->feature_dim = 0;
        return;
      }
      map = latest_map_;
    }

    response->features.data.clear();
    response->positions.clear();
    response->feature_dim = 0;

    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
      grid_map::Position center;
      map.getPosition(*it, center);
      std::vector<float> features = extract_features(map, center);
      if (features.empty()) {
        continue;
      }

      if (response->feature_dim == 0) {
        response->feature_dim = static_cast<int32_t>(features.size());
      }

      if (static_cast<int32_t>(features.size()) != response->feature_dim) {
        continue;
      }

      geometry_msgs::msg::Point position;
      position.x = center.x();
      position.y = center.y();
      position.z = 0.0;
      response->positions.push_back(position);
      response->features.data.insert(response->features.data.end(), features.begin(), features.end());
    }

    if (response->positions.empty()) {
      response->success.data = false;
      response->message.data = "No valid cell features found in current map";
      response->feature_dim = 0;
      response->features.data.clear();
      return;
    }

    response->success.data = true;
    response->message.data = "ok";
  }

  std::vector<float> extract_features(
    const grid_map::GridMap & map,
    const grid_map::Position & center) const
  {
    if (!map.exists("elevation") || !map.exists("color")) {
      return {};
    }
    if (!map.isInside(center)) {
      return {};
    }

    std::vector<float> features(6, 0.0F);
    grid_map::Position position;
    Eigen::Vector3f xyz;
    Eigen::Vector3f rgb;
    Eigen::Vector3f rgb_avg = Eigen::Vector3f::Zero();
    Eigen::Matrix3Xf points(3, 0);
    int color_count = 0;

    for (grid_map::CircleIterator it(map, center, feature_radius_); !it.isPastEnd(); ++it) {
      map.getPosition(*it, position);
      xyz(0) = static_cast<float>(position(0));
      xyz(1) = static_cast<float>(position(1));
      xyz(2) = map.at("elevation", *it);
      if (!xyz.array().isNaN().any()) {
        points.conservativeResize(Eigen::NoChange, points.cols() + 1);
        points.col(points.cols() - 1) = xyz;
      }

      grid_map::colorValueToVector(map.at("color", *it), rgb);
      if (rgb.array().isNaN().any()) {
        continue;
      }
      rgb_avg += rgb;
      ++color_count;
    }

    if (points.cols() < 3 || color_count == 0) {
      return {};
    }

    rgb_avg /= static_cast<float>(color_count);

    Eigen::Matrix3Xf centered = points.colwise() - points.rowwise().mean();
    Eigen::Matrix3f covariance = (centered * centered.transpose()) / static_cast<float>(points.cols() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
    if (eigensolver.info() != Eigen::Success) {
      return {};
    }

    Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
    const float largest = eigenvalues(2);
    if (largest <= 0.0F) {
      return {};
    }

    const float f_random = eigenvalues(0) / largest;
    const float f_plane = (eigenvalues(1) - eigenvalues(0)) / largest;
    const float f_line = (largest - eigenvalues(1)) / largest;

    features.at(0) = rgb_avg(0);
    features.at(1) = rgb_avg(1);
    features.at(2) = rgb_avg(2);
    features.at(3) = f_random;
    features.at(4) = f_plane;
    features.at(5) = f_line;

    return features;
  }

  std::string elevation_map_topic_;
  std::string feature_service_name_;
  std::string map_feature_service_name_;
  double feature_radius_;
  bool has_map_{false};
  grid_map::GridMap latest_map_;
  std::mutex map_mutex_;

  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr sub_;
  rclcpp::Service<soinn_slip_learner::srv::GetCellFeatures>::SharedPtr service_;
  rclcpp::Service<soinn_slip_learner::srv::GetMapFeatures>::SharedPtr map_service_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridmapFeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}
