#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <limits>

using namespace grid_map;

class FeaturemapPublisher : public rclcpp::Node
{
public:
  FeaturemapPublisher()
  : Node("featuremap_publisher")
  {
    // Parameters
    this->declare_parameter("map_length_x", 5.0);
    this->declare_parameter("map_length_y", 5.0);
    this->declare_parameter("resolution", 0.05);
    this->declare_parameter("pointcloud_topic", std::string("/zed/point_cloud/cloud_registered"));
    this->declare_parameter("frame_id", std::string("map"));

    const double map_length_x = this->get_parameter("map_length_x").as_double();
    const double map_length_y = this->get_parameter("map_length_y").as_double();
    const double resolution   = this->get_parameter("resolution").as_double();
    const std::string pointcloud_topic = this->get_parameter("pointcloud_topic").as_string();
    const std::string frame_id = this->get_parameter("frame_id").as_string();

    // Initialize grid map with elevation, color, and count layers
    map_ = GridMap({"elevation", "r", "g", "b", "count"});
    map_.setFrameId(frame_id);
    map_.setGeometry(Length(map_length_x, map_length_y), resolution);

    RCLCPP_INFO(this->get_logger(),
      "Created map with size %f x %f m (%i x %i cells).",
      map_.getLength().x(), map_.getLength().y(),
      map_.getSize()(0), map_.getSize()(1));

    publisher_ = this->create_publisher<grid_map_msgs::msg::GridMap>("grid_map", 1);
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic, 1,
      std::bind(&FeaturemapPublisher::pointCloudCallback, this, std::placeholders::_1));
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert ROS PointCloud2 to PCL
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *cloud);

    // Reset accumulation layers to zero
    map_["elevation"].setZero();
    map_["r"].setZero();
    map_["g"].setZero();
    map_["b"].setZero();
    map_["count"].setZero();

    // Accumulate point values into the corresponding grid cells
    for (const auto & point : cloud->points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }

      Index index;
      if (!map_.getIndex(Position(point.x, point.y), index)) {
        continue;
      }

      map_.at("elevation", index) += point.z;
      map_.at("r", index) += static_cast<float>(point.r);
      map_.at("g", index) += static_cast<float>(point.g);
      map_.at("b", index) += static_cast<float>(point.b);
      map_.at("count", index) += 1.0f;
    }

    // Compute per-cell mean; mark empty cells as NaN
    for (GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      const float count = map_.at("count", *it);
      if (count > 0.0f) {
        map_.at("elevation", *it) /= count;
        map_.at("r", *it) /= count;
        map_.at("g", *it) /= count;
        map_.at("b", *it) /= count;
      } else {
        map_.at("elevation", *it) = std::numeric_limits<float>::quiet_NaN();
        map_.at("r", *it) = std::numeric_limits<float>::quiet_NaN();
        map_.at("g", *it) = std::numeric_limits<float>::quiet_NaN();
        map_.at("b", *it) = std::numeric_limits<float>::quiet_NaN();
      }
    }

    // Publish the populated grid map
    map_.setTimestamp(rclcpp::Time(msg->header.stamp).nanoseconds());
    auto out_msg = grid_map::GridMapRosConverter::toMessage(map_);
    publisher_->publish(std::move(*out_msg));

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Grid map published from point cloud with %zu points.", cloud->size());
  }

  GridMap map_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeaturemapPublisher>());
  rclcpp::shutdown();
  return 0;
}


