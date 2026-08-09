#include <ad_lidar_perception/clustering/adaptive_euclidean_cluster.hpp>
#include <ad_lidar_perception/clustering/adaptive_euclidean_cluster_node.hpp>

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ad_lidar_perception::clustering
{

class AdaptiveEuclideanClusterNode::Impl
{
public:
  explicit Impl(AdaptiveClusterConfig config)
  : clusterer(std::move(config)) {}

  AdaptiveEuclideanCluster clusterer;
};

AdaptiveEuclideanClusterNode::AdaptiveEuclideanClusterNode(
  const rclcpp::NodeOptions & options)
: Node("ad_adaptive_euclidean_cluster", options)
{
  input_topic_ = declare_parameter<std::string>(
    "input_topic", "/ad/perception/lidar/nonground_finite");
  objects_topic_ = declare_parameter<std::string>(
    "objects_topic", "/ad/perception/objects/detected");
  clusters_topic_ = declare_parameter<std::string>(
    "clusters_topic", "/ad/perception/lidar/clusters");

  AdaptiveClusterConfig config;
  config.near_tolerance_m = declare_parameter<double>("near_tolerance_m", 0.45);
  config.far_tolerance_m = declare_parameter<double>("far_tolerance_m", 1.60);
  config.far_range_m = declare_parameter<double>("far_range_m", 45.0);
  config.minimum_points_far_range_m =
    declare_parameter<double>("minimum_points_far_range_m", 45.0);
  const int near_minimum_points =
    declare_parameter<int>("near_min_cluster_size", 5);
  const int far_minimum_points =
    declare_parameter<int>("far_min_cluster_size", 2);
  const int maximum_points = declare_parameter<int>("max_cluster_size", 20000);
  if (near_minimum_points <= 0 || far_minimum_points <= 0 ||
    maximum_points <= 0)
  {
    throw std::invalid_argument("cluster point counts must be positive");
  }
  config.near_minimum_points =
    static_cast<std::size_t>(near_minimum_points);
  config.far_minimum_points = static_cast<std::size_t>(far_minimum_points);
  config.maximum_points = static_cast<std::size_t>(maximum_points);
  config.use_height = declare_parameter<bool>("use_height", false);
  impl_ = std::make_unique<Impl>(config);

  maximum_dynamic_object_diagonal_m_ = declare_parameter<double>(
    "maximum_dynamic_object_diagonal_m", 12.0);
  (void)is_dynamic_component(
    0.0, 0.0, maximum_dynamic_object_diagonal_m_);

  min_x_m_ = declare_parameter<double>("crop.min_x_m", -4.0);
  max_x_m_ = declare_parameter<double>("crop.max_x_m", 100.0);
  min_y_m_ = declare_parameter<double>("crop.min_y_m", -25.0);
  max_y_m_ = declare_parameter<double>("crop.max_y_m", 25.0);
  min_z_m_ = declare_parameter<double>("crop.min_z_m", -1.0);
  max_z_m_ = declare_parameter<double>("crop.max_z_m", 3.0);
  if (min_x_m_ >= max_x_m_ || min_y_m_ >= max_y_m_ || min_z_m_ >= max_z_m_) {
    throw std::invalid_argument(
            "adaptive clustering crop bounds must increase");
  }

  objects_publisher_ =
    create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
    objects_topic_, 10);
  clusters_publisher_ =
    create_publisher<sensor_msgs::msg::PointCloud2>(clusters_topic_, 10);
  subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, rclcpp::SensorDataQoS(),
    std::bind(
      &AdaptiveEuclideanClusterNode::on_cloud, this,
      std::placeholders::_1));
}

AdaptiveEuclideanClusterNode::~AdaptiveEuclideanClusterNode() = default;

void AdaptiveEuclideanClusterNode::on_cloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
{
  std::vector<Point3> points;
  try {
    sensor_msgs::PointCloud2ConstIterator<float> x_iterator(*message, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_iterator(*message, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_iterator(*message, "z");
    points.reserve(static_cast<std::size_t>(message->width) * message->height);
    for (; x_iterator != x_iterator.end();
      ++x_iterator, ++y_iterator, ++z_iterator)
    {
      const double x = *x_iterator;
      const double y = *y_iterator;
      const double z = *z_iterator;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        x < min_x_m_ || x > max_x_m_ || y < min_y_m_ || y > max_y_m_ ||
        z < min_z_m_ || z > max_z_m_)
      {
        continue;
      }
      points.push_back({x, y, z});
    }
  } catch (const std::runtime_error & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Skipping invalid point cloud: %s", error.what());
    return;
  }

  const auto clusters = impl_->clusterer.cluster(points);
  autoware_perception_msgs::msg::DetectedObjects objects;
  objects.header = message->header;

  sensor_msgs::msg::PointCloud2 debug_cloud;
  debug_cloud.header = message->header;
  sensor_msgs::PointCloud2Modifier modifier(debug_cloud);
  modifier.setPointCloud2Fields(
    4,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
  std::size_t clustered_point_count = 0;
  for (const auto & cluster : clusters) {
    clustered_point_count += cluster.size();
  }
  modifier.resize(clustered_point_count);
  sensor_msgs::PointCloud2Iterator<float> debug_x(debug_cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> debug_y(debug_cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> debug_z(debug_cloud, "z");
  sensor_msgs::PointCloud2Iterator<float> debug_intensity(debug_cloud,
    "intensity");

  for (std::size_t cluster_index = 0; cluster_index < clusters.size();
    ++cluster_index)
  {
    const auto & cluster = clusters[cluster_index];
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (const std::size_t point_index : cluster) {
      const auto & point = points[point_index];
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      min_z = std::min(min_z, point.z);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
      max_z = std::max(max_z, point.z);
      *debug_x = static_cast<float>(point.x);
      *debug_y = static_cast<float>(point.y);
      *debug_z = static_cast<float>(point.z);
      *debug_intensity = static_cast<float>(cluster_index + 1U);
      ++debug_x;
      ++debug_y;
      ++debug_z;
      ++debug_intensity;
    }

    const double dimension_x = std::max(max_x - min_x, 0.10);
    const double dimension_y = std::max(max_y - min_y, 0.10);
    const double dimension_z = std::max(max_z - min_z, 0.10);
    if (!is_dynamic_component(
        dimension_x, dimension_y, maximum_dynamic_object_diagonal_m_))
    {
      continue;
    }

    autoware_perception_msgs::msg::DetectedObject object;
    object.existence_probability = 1.0F;
    autoware_perception_msgs::msg::ObjectClassification classification;
    classification.label =
      autoware_perception_msgs::msg::ObjectClassification::UNKNOWN;
    classification.probability = 1.0F;
    object.classification.push_back(classification);
    object.kinematics.pose_with_covariance.pose.position.x =
      0.5 * (min_x + max_x);
    object.kinematics.pose_with_covariance.pose.position.y =
      0.5 * (min_y + max_y);
    object.kinematics.pose_with_covariance.pose.position.z =
      0.5 * (min_z + max_z);
    object.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
    object.kinematics.orientation_availability =
      autoware_perception_msgs::msg::DetectedObjectKinematics::UNAVAILABLE;
    object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object.shape.dimensions.x = dimension_x;
    object.shape.dimensions.y = dimension_y;
    object.shape.dimensions.z = dimension_z;
    objects.objects.push_back(std::move(object));
  }

  debug_cloud.is_dense = true;
  objects_publisher_->publish(std::move(objects));
  clusters_publisher_->publish(std::move(debug_cloud));
}

} // namespace ad_lidar_perception::clustering
