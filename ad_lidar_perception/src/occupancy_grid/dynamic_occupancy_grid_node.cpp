#include "ad_lidar_perception/occupancy_grid/dynamic_occupancy_grid_node.hpp"

#include "ad_lidar_perception/occupancy_grid/dynamic_grid_builder.hpp"
#include "ad_lidar_perception/occupancy_grid/exact_stamp_pairer.hpp"

#include <ad_interfaces/msg/predicted_object.hpp>
#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <ad_interfaces/msg/predicted_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{
namespace
{

constexpr double kQuaternionTolerance = 1.0e-6;
constexpr double kCovarianceSymmetryTolerance = 1.0e-9;
constexpr std::size_t kMaximumPredictionStatesPerObject = 4096U;

bool positive_stamp(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.nanosec < 1'000'000'000U &&
         (stamp.sec > 0 || (stamp.sec == 0 && stamp.nanosec > 0U));
}

std::int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  if (!positive_stamp(stamp)) {
    throw std::invalid_argument("message stamp must be positive");
  }
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

bool finite_probability(const double probability)
{
  return std::isfinite(probability) &&
         probability >= 0.0 && probability <= 1.0;
}

void validate_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  if (!std::isfinite(quaternion.x) || !std::isfinite(quaternion.y) ||
    !std::isfinite(quaternion.z) || !std::isfinite(quaternion.w))
  {
    throw std::invalid_argument("predicted pose quaternion must be finite");
  }
  const double norm_squared =
    quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w;
  if (!std::isfinite(norm_squared) ||
    std::abs(norm_squared - 1.0) > kQuaternionTolerance)
  {
    throw std::invalid_argument("predicted pose quaternion must be normalized");
  }
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  validate_quaternion(quaternion);
  const double sine =
    2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double cosine =
    1.0 - 2.0 * (quaternion.y * quaternion.y +
    quaternion.z * quaternion.z);
  const double yaw = std::atan2(sine, cosine);
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("predicted pose yaw must be finite");
  }
  return yaw;
}

std::int64_t duration_ns(const builtin_interfaces::msg::Duration & duration)
{
  if (duration.sec < 0 || duration.nanosec >= 1'000'000'000U) {
    throw std::invalid_argument("prediction horizon must be nonnegative");
  }
  return static_cast<std::int64_t>(duration.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(duration.nanosec);
}

struct Covariance2
{
  double xx;
  double xy;
  double yy;
};

Covariance2 covariance_xy(
  const geometry_msgs::msg::PoseWithCovariance & pose)
{
  const double xx = pose.covariance[0];
  const double xy = pose.covariance[1];
  const double yx = pose.covariance[6];
  const double yy = pose.covariance[7];
  if (!std::isfinite(xx) || !std::isfinite(xy) ||
    !std::isfinite(yx) || !std::isfinite(yy))
  {
    throw std::invalid_argument("predicted pose covariance must be finite");
  }
  const double scale =
    std::max({1.0, std::abs(xy), std::abs(yx)});
  if (std::abs(xy - yx) > kCovarianceSymmetryTolerance * scale) {
    throw std::invalid_argument("predicted pose covariance must be symmetric");
  }
  return Covariance2{xx, (xy + yx) * 0.5, yy};
}

void validate_pose(const geometry_msgs::msg::PoseWithCovariance & pose)
{
  if (!std::isfinite(pose.pose.position.x) ||
    !std::isfinite(pose.pose.position.y) ||
    !std::isfinite(pose.pose.position.z))
  {
    throw std::invalid_argument("predicted pose position must be finite");
  }
  validate_quaternion(pose.pose.orientation);
  (void)covariance_xy(pose);
}

Covariance2 rotate_covariance(const Covariance2 & covariance, const double yaw)
{
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double cosine_squared = cosine * cosine;
  const double sine_squared = sine * sine;
  const double cross = sine * cosine;
  return Covariance2{
    cosine_squared * covariance.xx -
    2.0 * cross * covariance.xy +
    sine_squared * covariance.yy,
    cross * (covariance.xx - covariance.yy) +
    (cosine_squared - sine_squared) * covariance.xy,
    sine_squared * covariance.xx +
    2.0 * cross * covariance.xy +
    cosine_squared * covariance.yy};
}

std::size_t cell_count(
  const double minimum, const double maximum, const double resolution)
{
  if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
    !std::isfinite(resolution) || maximum <= minimum || resolution <= 0.0)
  {
    throw std::invalid_argument("invalid dynamic-grid extent");
  }
  const double cells = (maximum - minimum) / resolution;
  if (!std::isfinite(cells) || cells < 0.5 ||
    cells > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
  {
    throw std::invalid_argument("dynamic-grid extent is not representable");
  }
  const double rounded_cells = std::round(cells);
  const double tolerance =
    8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
    std::max(1.0, std::abs(cells));
  if (std::abs(cells - rounded_cells) > tolerance) {
    throw std::invalid_argument(
            "dynamic-grid extent must contain an integer number of cells");
  }
  return static_cast<std::size_t>(rounded_cells);
}

}  // namespace

class DynamicOccupancyGridNode : public rclcpp::Node
{
public:
  DynamicOccupancyGridNode()
  : Node("ad_dynamic_occupancy_grid"),
    target_frame_(
      declare_parameter<std::string>("target_frame", "base_link")),
    source_frame_(declare_parameter<std::string>("source_frame", "odom")),
    transform_timeout_sec_(
      declare_parameter<double>("transform_timeout_sec", 0.05)),
    prediction_timeout_sec_(
      declare_parameter<double>("prediction_timeout_sec", 0.50)),
    stale_check_period_sec_(
      declare_parameter<double>("stale_check_period_sec", 0.10)),
    road_gate_enabled_(
      declare_parameter<bool>("road_gate.enabled", false)),
    road_gate_maximum_pending_messages_(
      declare_parameter<int>("road_gate.maximum_pending_messages", 8)),
    x_max_m_(declare_parameter<double>("x_max", 100.0)),
    y_max_m_(declare_parameter<double>("y_max", 10.0)),
    geometry_{
      declare_parameter<double>("x_min", -4.0),
      declare_parameter<double>("y_min", -10.0),
      static_cast<double>(static_cast<float>(
        declare_parameter<double>("resolution", 0.1))),
      0U,
      0U},
    config_{
      declare_parameter<double>("covariance_sigma", 2.0),
      declare_parameter<double>("minimum_inflation_m", 0.20),
      static_cast<std::int8_t>(0),
      0U},
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    const auto occupied_cost =
      declare_parameter<std::int64_t>("occupied_cost", 100);
    const auto maximum_cells =
      declare_parameter<std::int64_t>("maximum_cells_per_object", 20000);
    if (occupied_cost < 1 || occupied_cost > 100 ||
      maximum_cells <= 0 ||
      static_cast<std::uint64_t>(maximum_cells) >
      std::numeric_limits<std::size_t>::max())
    {
      throw std::invalid_argument("invalid dynamic occupancy cost or guard");
    }
    config_.occupied_cost = static_cast<std::int8_t>(occupied_cost);
    config_.maximum_cells_per_object =
      static_cast<std::size_t>(maximum_cells);
    geometry_.width =
      cell_count(geometry_.x_min_m, x_max_m_, geometry_.resolution_m);
    geometry_.height =
      cell_count(geometry_.y_min_m, y_max_m_, geometry_.resolution_m);

    if (target_frame_ != "base_link" || source_frame_ != "odom") {
      throw std::invalid_argument(
              "dynamic occupancy requires source_frame=odom and target_frame=base_link");
    }
    if (!std::isfinite(transform_timeout_sec_) ||
      !std::isfinite(prediction_timeout_sec_) ||
      !std::isfinite(stale_check_period_sec_) ||
      transform_timeout_sec_ <= 0.0 ||
      prediction_timeout_sec_ <= 0.0 ||
      stale_check_period_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "dynamic occupancy timing is invalid");
    }
    // Validate the complete pure configuration before creating graph endpoints.
    (void)build_dynamic_grid(geometry_, {}, config_);

    const auto predicted_topic = declare_parameter<std::string>(
      "topics.predicted_objects", "/ad/perception/objects/predicted");
    const auto drivable_mask_topic = declare_parameter<std::string>(
      "topics.drivable_mask", "/ad/planning/drivable_mask");
    const auto dynamic_topic = declare_parameter<std::string>(
      "topics.dynamic_grid", "/ad/perception/occupancy/dynamic");
    if (predicted_topic.empty() || dynamic_topic.empty() ||
      get_node_topics_interface()->resolve_topic_name(predicted_topic) ==
      get_node_topics_interface()->resolve_topic_name(dynamic_topic))
    {
      throw std::invalid_argument(
              "predicted-object and dynamic-grid topics must be distinct");
    }
    if (road_gate_maximum_pending_messages_ <= 0 ||
      road_gate_maximum_pending_messages_ > 1024)
    {
      throw std::invalid_argument(
              "road_gate.maximum_pending_messages must be in [1, 1024]");
    }
    if (road_gate_enabled_ && drivable_mask_topic.empty()) {
      throw std::invalid_argument(
              "enabled road gate requires topics.drivable_mask");
    }

    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      dynamic_topic, rclcpp::SensorDataQoS());
    if (road_gate_enabled_) {
      road_gate_pairer_ = std::make_unique<RoadGatePairer>(
        static_cast<std::size_t>(road_gate_maximum_pending_messages_));
      drivable_mask_subscription_ =
        create_subscription<nav_msgs::msg::OccupancyGrid>(
        drivable_mask_topic, rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr input) {
          on_drivable_mask(std::move(input));
        });
    }
    subscription_ =
      create_subscription<ad_interfaces::msg::PredictedObjectArray>(
      predicted_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      [this](
        ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input)
      {
        on_prediction_message(std::move(input));
      });
    stale_timer_ = rclcpp::create_timer(
      this, get_clock(),
      rclcpp::Duration::from_seconds(stale_check_period_sec_),
      [this]() {on_stale_timer();});
  }

private:
  using RoadGatePairer = ExactStampPairer<
    ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr,
    std::vector<std::int8_t>>;

  DynamicBox transformed_box(
    const geometry_msgs::msg::PoseWithCovariance & input_pose,
    const double length,
    const double width,
    const geometry_msgs::msg::TransformStamped & transform) const
  {
    validate_pose(input_pose);
    validate_quaternion(transform.transform.rotation);

    geometry_msgs::msg::PoseStamped source_pose;
    source_pose.header.frame_id = source_frame_;
    source_pose.header.stamp = transform.header.stamp;
    source_pose.pose = input_pose.pose;
    geometry_msgs::msg::PoseStamped target_pose;
    tf2::doTransform(source_pose, target_pose, transform);
    if (!std::isfinite(target_pose.pose.position.x) ||
      !std::isfinite(target_pose.pose.position.y) ||
      !std::isfinite(target_pose.pose.position.z))
    {
      throw std::invalid_argument("transformed predicted pose must be finite");
    }
    const double transform_yaw =
      yaw_from_quaternion(transform.transform.rotation);
    const auto covariance =
      rotate_covariance(covariance_xy(input_pose), transform_yaw);
    return DynamicBox{
      target_pose.pose.position.x,
      target_pose.pose.position.y,
      yaw_from_quaternion(target_pose.pose.orientation),
      length,
      width,
      covariance.xx,
      covariance.xy,
      covariance.yy};
  }

  std::vector<DynamicBox> boxes_from_message(
    const ad_interfaces::msg::PredictedObjectArray & input,
    const geometry_msgs::msg::TransformStamped & transform) const
  {
    std::vector<DynamicBox> boxes;
    if (input.objects.size() > boxes.max_size()) {
      throw std::length_error("predicted-object set is not representable");
    }
    boxes.reserve(input.objects.size());
    for (const auto & object : input.objects) {
      if (!finite_probability(object.existence_probability) ||
        !finite_probability(object.classification_probability) ||
        object.classification >
        ad_interfaces::msg::PredictedObject::PEDESTRIAN)
      {
        throw std::invalid_argument(
                "predicted-object probabilities or classification are invalid");
      }
      if (!std::isfinite(object.dimensions.x) ||
        !std::isfinite(object.dimensions.y) ||
        !std::isfinite(object.dimensions.z) ||
        object.dimensions.x <= 0.0 ||
        object.dimensions.y <= 0.0 ||
        object.dimensions.z <= 0.0)
      {
        throw std::invalid_argument(
                "predicted-object dimensions must be finite and positive");
      }
      if (object.states.empty()) {
        throw std::invalid_argument(
                "predicted object must contain at least one future horizon");
      }
      if (object.states.size() > kMaximumPredictionStatesPerObject) {
        throw std::length_error(
                "predicted-object horizon count is not representable");
      }
      boxes.push_back(
        transformed_box(
          object.initial_pose, object.dimensions.x, object.dimensions.y,
          transform));
      std::int64_t previous_horizon = 0;
      for (const auto & state : object.states) {
        const auto horizon = duration_ns(state.time_from_start);
        if (horizon <= previous_horizon) {
          throw std::invalid_argument(
                  "predicted-object horizons must be strictly increasing");
        }
        validate_pose(state.pose);
        previous_horizon = horizon;
      }
    }
    // OccupancyGrid has no time axis. Rasterizing every future keyframe would
    // make the complete trajectory occupied at every DWA rollout instant.
    // Keep only the current footprint here; the planner consumes the original
    // time-indexed prediction for swept collision checks and visualization.
    return boxes;
  }

  nav_msgs::msg::OccupancyGrid make_grid(
    const builtin_interfaces::msg::Time & stamp,
    std::vector<std::int8_t> data) const
  {
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = stamp;
    grid.header.frame_id = target_frame_;
    grid.info.resolution = static_cast<float>(geometry_.resolution_m);
    grid.info.width = static_cast<std::uint32_t>(geometry_.width);
    grid.info.height = static_cast<std::uint32_t>(geometry_.height);
    grid.info.origin.position.x = geometry_.x_min_m;
    grid.info.origin.position.y = geometry_.y_min_m;
    grid.info.origin.orientation.w = 1.0;
    grid.data = std::move(data);
    return grid;
  }

  std::vector<std::int8_t> convert_drivable_mask(
    const nav_msgs::msg::OccupancyGrid & input) const
  {
    (void)stamp_nanoseconds(input.header.stamp);
    if (input.header.frame_id != target_frame_) {
      throw std::invalid_argument(
              "drivable-mask frame must match target_frame");
    }
    const auto & origin = input.info.origin;
    if (!std::isfinite(origin.position.x) ||
      !std::isfinite(origin.position.y) ||
      !std::isfinite(origin.position.z) ||
      !std::isfinite(origin.orientation.x) ||
      !std::isfinite(origin.orientation.y) ||
      !std::isfinite(origin.orientation.z) ||
      !std::isfinite(origin.orientation.w) ||
      origin.position.x != geometry_.x_min_m ||
      origin.position.y != geometry_.y_min_m ||
      origin.position.z != 0.0 ||
      origin.orientation.x != 0.0 ||
      origin.orientation.y != 0.0 ||
      origin.orientation.z != 0.0 ||
      origin.orientation.w != 1.0 ||
      static_cast<double>(input.info.resolution) != geometry_.resolution_m ||
      static_cast<std::size_t>(input.info.width) != geometry_.width ||
      static_cast<std::size_t>(input.info.height) != geometry_.height ||
      input.data.size() != geometry_.width * geometry_.height)
    {
      throw std::invalid_argument(
              "drivable-mask geometry must exactly match dynamic grid");
    }
    if (std::any_of(
        input.data.begin(), input.data.end(),
        [](const std::int8_t value) {
          return value < -1 || value > 100;
        }))
    {
      throw std::invalid_argument("drivable mask has an invalid cell value");
    }
    return input.data;
  }

  void publish_clear(const rclcpp::Time & stamp)
  {
    publisher_->publish(
      make_grid(
        static_cast<builtin_interfaces::msg::Time>(stamp),
        build_dynamic_grid(geometry_, {}, config_)));
  }

  void invalidate_and_clear(
    const std::string & reason,
    const std::optional<rclcpp::Time> & clear_stamp = std::nullopt)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "clearing dynamic occupancy layer: %s", reason.c_str());
    last_valid_stamp_.reset();
    if (clear_stamp.has_value()) {
      publish_clear(*clear_stamp);
    } else if (latest_admissible_stamp_.has_value()) {
      publish_clear(*latest_admissible_stamp_);
    } else {
      publish_clear(get_clock()->now());
    }
  }

  void on_predictions(
    const ad_interfaces::msg::PredictedObjectArray & input,
    const std::vector<std::int8_t> * const mask)
  {
    std::optional<rclcpp::Time> admitted_stamp;
    try {
      if (input.header.frame_id != source_frame_) {
        throw std::invalid_argument(
                "predicted-object frame must be exactly odom");
      }
      if (!positive_stamp(input.header.stamp)) {
        throw std::invalid_argument(
                "predicted-object stamp must be positive");
      }
      const auto stamp = rclcpp::Time(input.header.stamp);
      const auto now = get_clock()->now();
      if (latest_admissible_stamp_.has_value() &&
        now < *latest_admissible_stamp_)
      {
        last_valid_stamp_.reset();
        latest_admissible_stamp_.reset();
      }
      if (stamp > now) {
        throw std::invalid_argument("predicted-object stamp is in the future");
      }
      if (now - stamp >
        rclcpp::Duration::from_seconds(prediction_timeout_sec_))
      {
        throw std::invalid_argument("predicted-object stamp is already stale");
      }
      if (latest_admissible_stamp_.has_value() &&
        stamp < *latest_admissible_stamp_)
      {
        throw std::invalid_argument("predicted-object stamp is out of order");
      }
      admitted_stamp = stamp;
      latest_admissible_stamp_ = stamp;
      const auto transform = tf_buffer_.lookupTransform(
        target_frame_, input.header.frame_id,
        rclcpp::Time(input.header.stamp),
        rclcpp::Duration::from_seconds(transform_timeout_sec_));
      const auto boxes = boxes_from_message(input, transform);
      std::vector<std::int8_t> data;
      if (mask == nullptr) {
        data = build_dynamic_grid(geometry_, boxes, config_);
      } else {
        const auto & drivable_mask = *mask;
        data = build_dynamic_grid(geometry_, boxes, config_, drivable_mask);
      }
      publisher_->publish(
        make_grid(
          input.header.stamp,
          std::move(data)));
      last_valid_stamp_ = stamp;
    } catch (const std::exception & error) {
      invalidate_and_clear(error.what(), admitted_stamp);
    }
  }

  void process_pair(std::optional<RoadGatePairer::Pair> pair)
  {
    if (pair.has_value()) {
      on_predictions(*pair->left, &pair->right);
    }
  }

  void on_prediction_message(
    ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input)
  {
    if (!road_gate_enabled_) {
      on_predictions(*input, nullptr);
      return;
    }
    try {
      if (input->header.frame_id != source_frame_) {
        throw std::invalid_argument(
                "predicted-object frame must be exactly odom");
      }
      const auto stamp_ns = stamp_nanoseconds(input->header.stamp);
      process_pair(
        road_gate_pairer_->add_left(stamp_ns, std::move(input)));
    } catch (const std::exception & error) {
      invalidate_and_clear(error.what());
    }
  }

  void on_drivable_mask(
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr input)
  {
    try {
      const auto stamp_ns = stamp_nanoseconds(input->header.stamp);
      auto mask = convert_drivable_mask(*input);
      process_pair(
        road_gate_pairer_->add_right(stamp_ns, std::move(mask)));
    } catch (const std::exception & error) {
      invalidate_and_clear(error.what());
    }
  }

  void on_stale_timer()
  {
    if (!last_valid_stamp_.has_value()) {
      return;
    }
    const auto now = get_clock()->now();
    if (now < *last_valid_stamp_) {
      latest_admissible_stamp_.reset();
      invalidate_and_clear("ROS clock rolled back", now);
      return;
    }
    if (now - *last_valid_stamp_ >
      rclcpp::Duration::from_seconds(prediction_timeout_sec_))
    {
      invalidate_and_clear("prediction timed out", now);
    }
  }

  std::string target_frame_;
  std::string source_frame_;
  double transform_timeout_sec_;
  double prediction_timeout_sec_;
  double stale_check_period_sec_;
  bool road_gate_enabled_;
  int road_gate_maximum_pending_messages_;
  double x_max_m_;
  double y_max_m_;
  GridGeometry geometry_;
  DynamicGridConfig config_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Subscription<
    ad_interfaces::msg::PredictedObjectArray>::SharedPtr subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
    drivable_mask_subscription_;
  std::unique_ptr<RoadGatePairer> road_gate_pairer_;
  rclcpp::TimerBase::SharedPtr stale_timer_;
  std::optional<rclcpp::Time> last_valid_stamp_;
  std::optional<rclcpp::Time> latest_admissible_stamp_;
};

std::shared_ptr<rclcpp::Node> make_dynamic_occupancy_grid_node()
{
  return std::make_shared<DynamicOccupancyGridNode>();
}

}  // namespace ad_lidar_perception::occupancy_grid
