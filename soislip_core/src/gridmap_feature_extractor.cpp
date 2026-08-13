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
    this->declare_parameter("ellipse_a", 0.0);
    this->declare_parameter("ellipse_b", 0.0);
    this->declare_parameter("robot_max_climbable_slope_deg", 27.0);

    this->get_parameter("elevation_map_topic", elevation_map_topic_);
    this->get_parameter("feature_service_name", feature_service_name_);
    this->get_parameter("map_feature_service_name", map_feature_service_name_);
    this->get_parameter("ellipse_a", ellipse_a_);
    this->get_parameter("ellipse_b", ellipse_b_);
    this->get_parameter("robot_max_climbable_slope_deg", robot_max_climbable_slope_deg_);

    sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
      elevation_map_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridmapFeatureExtractorNode::callback, this, std::placeholders::_1));

    map_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
      "/soislip/feature_map", 10);

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
    float nan = std::numeric_limits<float>::quiet_NaN();
    grid_map::GridMapRosConverter::fromMessage(*msg, new_map);
    std::scoped_lock<std::mutex> lock(map_mutex_);
    if (!has_map_) {
      RCLCPP_INFO(this->get_logger(), "Received first grid map");
      feature_map_ = new_map;
      feature_map_.add("seen", 0.0F);
      feature_map_.add("feature_slope", nan);
      feature_map_.add("feature_color", nan);
      has_map_ = true;
    }
    merged_map = new_map;
    merged_map.add("seen", 0.0F);
    merged_map.add("feature_slope", nan);
    merged_map.add("feature_color", nan);
    merged_map.addDataFrom(feature_map_, false, true, false, {"seen", "feature_slope", "feature_color"});
    feature_map_ = merged_map;
    auto fm_msg = grid_map::GridMapRosConverter::toMessage(feature_map_);
    map_pub_->publish(*fm_msg);
  }

  void handle_get_cell_features(
    const std::shared_ptr<soislip_interfaces::srv::GetCellFeatures::Request> request,
    std::shared_ptr<soislip_interfaces::srv::GetCellFeatures::Response> response)
  {
    const grid_map::Position center(
      static_cast<double>(request->position.x),
      static_cast<double>(request->position.y));
    const double rotation = static_cast<double>(request->rotation);

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

    std::vector<float> features = extract_features(map, center, rotation, true);
    if (features.empty()) {
      response->success.data = false;
      response->message.data = "Could not extract features at requested position";
      return;
    }

    response->features.data = features;
    response->success.data = true;
    response->message.data = "ok";

    Eigen::Vector3f rgb = lab_to_rgb(features.at(0), features.at(1));
    float color_value;
    grid_map::colorVectorToValue(rgb, color_value);
    {
      std::scoped_lock<std::mutex> lock(map_mutex_);
      feature_map_.atPosition("feature_color", center) = color_value;
      feature_map_.atPosition("feature_slope", center) = features.at(2);  // Store slope feature
      feature_map_.atPosition("seen", center) = 0.3F;  // Mark the center cell as seen
    }
  }

  void handle_get_map_features(
    const std::shared_ptr<soislip_interfaces::srv::GetMapFeatures::Request> request,
    std::shared_ptr<soislip_interfaces::srv::GetMapFeatures::Response> response)
  {
    double rotation = static_cast<double>(request->rotation);
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
      std::vector<float> features;
      map.getPosition(*it, center);
      features = extract_features(map, center, rotation);
      if (features.empty()) {
        continue;
      }
      if (response->feature_dim == 0) {
        // Set the feature dimension based on the first valid feature vector
        response->feature_dim = static_cast<int32_t>(features.size());
      }
      geometry_msgs::msg::Point position;
      position.x = center.x();
      position.y = center.y();
      position.z = 0.0;
      response->positions.push_back(position);
      response->features.data.insert(response->features.data.end(), features.begin(), features.end());
      Eigen::Vector3f rgb = lab_to_rgb(features.at(0), features.at(1));
      float color_value;
      grid_map::colorVectorToValue(rgb, color_value);
      {
        std::scoped_lock<std::mutex> lock(map_mutex_);
        feature_map_.atPosition("feature_color", center) = color_value;
        feature_map_.atPosition("feature_slope", center) = features.at(2);  // Store slope feature
      }
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
    const grid_map::Position & center,
    const double rotation,
    const bool markCells = false)
  {
    const int feature_dim = 3;  // a, b, slope_percent
    if (!map.exists("elevation") || !map.exists("color")) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Grid map does not contain required layers 'elevation' and 'color'");
        return {};
    }
    if (!map.isInside(center)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Requested position (%.2f, %.2f) is outside the grid map bounds", center.x(), center.y());
      return {};
    }
    std::vector<float> features(feature_dim, 0.0F);
    grid_map::Position position;
    Eigen::Vector3f xyz;
    Eigen::Vector3f rgb;
    Eigen::Vector3f color_sum = Eigen::Vector3f::Zero();
    Eigen::Matrix3Xf points(3, 0);
    int color_count = 0;
    // Use a radius that is at least 1.5 times the map resolution to ensure enough points are sampled
    double length_a = std::max(ellipse_a_, map.getResolution()*3.0);
    double length_b = std::max(ellipse_b_, map.getResolution()*3.0);
    if(markCells) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Extracting features at position (%.2f, %.2f) with ellipse axes (%.2f, %.2f) and rotation %.2f rad",
        center.x(), center.y(), length_a, length_b, rotation);
      }
    grid_map::Length length(length_a, length_b);
    for (grid_map::EllipseIterator it(map, center, length, rotation); !it.isPastEnd(); ++it) {
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
      if (markCells)
      {
        std::scoped_lock<std::mutex> lock(map_mutex_);
        feature_map_.atPosition("seen", position) = 0.3F;  // Mark the center cell as seen
      }
    }
    if (points.cols() < 3 || color_count == 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Not enough valid points in the neighborhood to compute features at position (%.2f, %.2f)",
        center.x(), center.y());
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
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Eigen decomposition failed at position (%.2f, %.2f)", center.x(), center.y());
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
    features.at(2) = slope_percent;
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

  static float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
  }

  static float linear_to_srgb(float c) {
    c = std::max(0.0f, c);
    if (c <= 0.0031308F) {
      return 12.92F * c;
    }
    return 1.055F * std::pow(c, 1.0F / 2.4F) - 0.055F;
  }

  static float lab_f_inv(float t) {
    constexpr float eps = 216.0F / 24389.0F;
    constexpr float k = 24389.0F / 27.0F;
    const float t3 = t * t * t;
    if (t3 > eps) {
      return t3;
    }
    return (116.0F * t - 16.0F) / k;
  }

  // Input: Lab a,b
  // Output: bright/saturated sRGB color in [0,1]
  static Eigen::Vector3f lab_to_rgb(
    float a,
    float b,
    float L = 75.0F,          // brightness target (0..100)
    float target_chroma = 55.0F)  // saturation target in Lab
  {
    // Normalize chroma to a controlled saturation level.
    const float chroma = std::sqrt(a * a + b * b);
    if (chroma > 1e-6F) {
      const float s = target_chroma / chroma;
      a *= s;
      b *= s;
    }

    // Lab -> XYZ (D65)
    constexpr float Xn = 0.95047F;
    constexpr float Yn = 1.00000F;
    constexpr float Zn = 1.08883F;

    const float fy = (L + 16.0F) / 116.0F;
    const float fx = fy + a / 500.0F;
    const float fz = fy - b / 200.0F;

    const float X = Xn * lab_f_inv(fx);
    const float Y = Yn * lab_f_inv(fy);
    const float Z = Zn * lab_f_inv(fz);

    // XYZ -> linear sRGB
    float r_lin =  3.2404542F * X - 1.5371385F * Y - 0.4985314F * Z;
    float g_lin = -0.9692660F * X + 1.8760108F * Y + 0.0415560F * Z;
    float b_lin =  0.0556434F * X - 0.2040259F * Y + 1.0572252F * Z;

    // Brighten while preserving hue as much as possible.
    const float max_lin = std::max(r_lin, std::max(g_lin, b_lin));
    if (max_lin > 1e-6F && max_lin < 1.0F) {
      const float gain = 1.0F / max_lin;
      r_lin *= gain;
      g_lin *= gain;
      b_lin *= gain;
    }

    // Gamma encode + clamp to [0,1]
    const float r = clamp01(linear_to_srgb(r_lin));
    const float g = clamp01(linear_to_srgb(g_lin));
    const float b_out = clamp01(linear_to_srgb(b_lin));

    return Eigen::Vector3f(r, g, b_out);
  }




  std::string elevation_map_topic_;
  std::string feature_service_name_;
  std::string map_feature_service_name_;
  double ellipse_a_; // semi-major axis of the ellipse for feature extraction
  double ellipse_b_; // semi-minor axis of the ellipse for feature extraction
  float robot_max_climbable_slope_deg_;
  bool has_map_{false};
  grid_map::GridMap feature_map_; // latest received map
  grid_map::GridMap internal_map_; // internal map to store features and mark seen cells
  std::mutex map_mutex_;

  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr map_pub_;
  rclcpp::Service<soislip_interfaces::srv::GetCellFeatures>::SharedPtr service_;
  rclcpp::Service<soislip_interfaces::srv::GetMapFeatures>::SharedPtr map_service_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridmapFeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}
