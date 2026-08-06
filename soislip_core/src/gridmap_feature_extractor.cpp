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

#include "soislip_interfaces/srv/get_cell_features.hpp"
#include "soislip_interfaces/srv/get_map_features.hpp"

class GridmapFeatureExtractorNode : public rclcpp::Node {
public:
  GridmapFeatureExtractorNode()
  : Node("gridmap_feature_extractor_node") {
    this->declare_parameter("elevation_map_topic", "/elevation_map");
    this->declare_parameter("feature_service_name", "get_cell_features");
    this->declare_parameter("map_feature_service_name", "get_map_features");
    this->declare_parameter("feature_radius", 0.0);
    this->declare_parameter("robot_max_climbable_slope_deg", 27.0);

    this->get_parameter("elevation_map_topic", elevation_map_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("map_feature_service_name", map_feature_service_name_);
    this->get_parameter("feature_radius", feature_radius_);
    this->get_parameter("robot_max_climbable_slope_deg", robot_max_climbable_slope_deg_);

    sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
      elevation_map_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridmapFeatureExtractorNode::callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
      "feature_map",
      rclcpp::SensorDataQoS());

    service_ = this->create_service<soislip_interfaces::srv::GetCellFeatures>(
      feature_service_name_,
      std::bind(
        &GridmapFeatureExtractorNode::handle_get_cell_features,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    map_service_ = this->create_service<soislip_interfaces::srv::GetMapFeatures>(
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
    grid_map::GridMap new_map;
    grid_map::GridMap merged_map;
    grid_map::GridMapRosConverter::fromMessage(*msg, new_map);
    std::scoped_lock<std::mutex> lock(map_mutex_);
    if (!has_map_) {
      RCLCPP_INFO(this->get_logger(), "Received first grid map");
      feature_map_ = new_map;
      feature_map_.add("seen", 0.0F);
      feature_map_.add("feature_slope", 0.0F);
      feature_map_.add("feature_color", 0.0F);
      has_map_ = true;
    }
    merged_map = feature_map_;
    // Update the geometry of internal_map_ to match the new map
    merged_map.setGeometry(new_map.getLength(), new_map.getResolution(), new_map.getPosition());
    // Merge the new map into the internal_map_ to keep track of seen cells
    merged_map.addDataFrom(new_map, true, true, true);

    feature_map_ = merged_map;
    pub_->publish(grid_map::GridMapRosConverter::toMessage(feature_map_));
  }

  void handle_get_cell_features(
    const std::shared_ptr<soislip_interfaces::srv::GetCellFeatures::Request> request,
    std::shared_ptr<soislip_interfaces::srv::GetCellFeatures::Response> response)
  {
    grid_map::GridMap map;
    {
      std::scoped_lock<std::mutex> lock(map_mutex_);
      if (!has_map_) {
        response->success.data = false;
        response->message.data = "No grid map received yet";
        return;
      }
      map = feature_map_;
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
    const std::shared_ptr<soislip_interfaces::srv::GetMapFeatures::Request> request,
    std::shared_ptr<soislip_interfaces::srv::GetMapFeatures::Response> response)
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
      map = feature_map_;
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
    Eigen::Vector3f color_sum = Eigen::Vector3f::Zero();
    Eigen::Matrix3Xf points(3, 0);
    int color_count = 0;
    // Use a radius that is at least 1.5 times the map resolution to ensure enough points are sampled
    double radius = std::max(feature_radius_, map.getResolution()*1.5);

    for (grid_map::CircleIterator it(map, center, radius); !it.isPastEnd(); ++it) {
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
      // Normalized RGB values to avoid biasing the average color towards brighter colors
      // color_sum += rgb / rgb.sum();

      // or Lab color
      const Eigen::Vector3f lab = rgb_to_lab(rgb);
      color_sum += lab;

      ++color_count;

      // Mark seen cell in map
      // map.at("color", *it) = 10000000000.0F;
    }
    if (points.cols() < 3 || color_count == 0) {
      return {};
    }

    // Compute the average color and normalize it by the number of valid color points
    // features.at(0) = color_sum(0) / static_cast<float>(color_count); // r
    // features.at(1) = color_sum(1) / static_cast<float>(color_count); // g
    // features.at(2) = color_sum(2) / static_cast<float>(color_count); // b
    features.at(0) = color_sum(1) / static_cast<float>(color_count); // a
    features.at(1) = color_sum(2) / static_cast<float>(color_count); // b

    Eigen::Matrix3Xf centered = points.colwise() - points.rowwise().mean();
    Eigen::Matrix3f covariance = (centered * centered.transpose()) / static_cast<float>(points.cols() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
    if (eigensolver.info() != Eigen::Success) {
      return {};
    }

    // MY FEATURES
    // Smallest eigenvalue eigenvector = local plane normal
    Eigen::Vector3f normal = eigensolver.eigenvectors().col(0).normalized();
    if (normal(2) < 0.0F) {
      normal = -normal;  // keep it pointing upward
    }

    const float horizontal = std::sqrt(normal(0) * normal(0) + normal(1) * normal(1));
    const float slope_rad = std::atan2(horizontal, std::fabs(normal(2)));
    const float slope_deg = slope_rad * 180.0F / static_cast<float>(M_PI);
    const float slope_percent = slope_deg / robot_max_climbable_slope_deg_;

    
    // Optional roughness proxy:
    const Eigen::Vector3f evals = eigensolver.eigenvalues();
    const float roughness = evals(0);  // smaller is smoother

    features.at(3) = slope_percent;

    
    // PRAGR'S FEATURES
    // Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
    // const float largest = eigenvalues(2);
    // if (largest <= 0.0F) {
    //   return {};
    // }

    // const float f_random = eigenvalues(0) / largest;
    // const float f_plane = (eigenvalues(1) - eigenvalues(0)) / largest;
    // const float f_line = (largest - eigenvalues(1)) / largest;

    // features.at(3) = f_random;
    // features.at(4) = f_plane;
    // features.at(5) = f_line;

    return features;
  }

  static float srgb_to_linear(float c) {
    if (c <= 0.04045F) {
      return c / 12.92F;
    }
    return std::pow((c + 0.055F) / 1.055F, 2.4F);
  }

  static Eigen::Vector3f rgb_to_lab(const Eigen::Vector3f & rgb_in) {
    // This function expects [0,1] input range.
    Eigen::Vector3f rgb = rgb_in;

    const float r = srgb_to_linear(std::clamp(rgb(0), 0.0F, 1.0F));
    const float g = srgb_to_linear(std::clamp(rgb(1), 0.0F, 1.0F));
    const float b = srgb_to_linear(std::clamp(rgb(2), 0.0F, 1.0F));

    // sRGB D65 -> XYZ
    const float X = 0.4124564F * r + 0.3575761F * g + 0.1804375F * b;
    const float Y = 0.2126729F * r + 0.7151522F * g + 0.0721750F * b;
    const float Z = 0.0193339F * r + 0.1191920F * g + 0.9503041F * b;

    // D65 white
    constexpr float Xn = 0.95047F;
    constexpr float Yn = 1.00000F;
    constexpr float Zn = 1.08883F;

    auto f = [](float t) -> float {
      constexpr float eps = 216.0F / 24389.0F;
      constexpr float k = 24389.0F / 27.0F;
      if (t > eps) {
        return std::cbrt(t);
      }
      return (k * t + 16.0F) / 116.0F;
    };

    const float fx = f(X / Xn);
    const float fy = f(Y / Yn);
    const float fz = f(Z / Zn);

    const float L = 116.0F * fy - 16.0F;
    const float a = 500.0F * (fx - fy);
    const float b_lab = 200.0F * (fy - fz);
    return Eigen::Vector3f(L, a, b_lab);
  }



  std::string elevation_map_topic_;
  std::string feature_service_name_;
  std::string map_feature_service_name_;
  double feature_radius_;
  float robot_max_climbable_slope_deg_;
  bool has_map_{false};
  grid_map::GridMap feature_map_; // latest received map
  grid_map::GridMap internal_map_; // internal map to store features and mark seen cells
  std::mutex map_mutex_;

  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_;
  rclcpp::Service<soislip_interfaces::srv::GetCellFeatures>::SharedPtr service_;
  rclcpp::Service<soislip_interfaces::srv::GetMapFeatures>::SharedPtr map_service_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridmapFeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}
