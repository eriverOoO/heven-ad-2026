#ifndef AD_LOCALIZATION__GNSS_SHADOW__LOCALIZATION_HANDOFF_HPP_
#define AD_LOCALIZATION__GNSS_SHADOW__LOCALIZATION_HANDOFF_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"

namespace ad_localization
{

struct Point2
{
  double x{0.0};
  double y{0.0};
};

enum class Backend
{
  kGnssImu,
  kFastlio,
};

enum class HandoffPhase
{
  kGnssApproach,
  kFastlioReady,
  kFastlioActive,
  kGnssRecovery,
  kGnssFinish,
};

struct HandoffConfig
{
  Point2 entry_xy;
  Point2 exit_xy;
  double prewarm_radius_m{20.0};
  double entry_switch_radius_m{8.0};
  double exit_switch_radius_m{20.0};
  double source_timeout_sec{0.5};
  double maximum_position_disagreement_m{2.0};
  double maximum_yaw_disagreement_rad{0.2};
  double blend_duration_sec{2.0};
  std::string reference_frame{"odom"};
  std::string base_frame{"base_link"};
};

struct HandoffUpdate
{
  std::optional<nav_msgs::msg::Odometry> canonical_odometry;
  std::optional<geometry_msgs::msg::PoseStamped> fastlio_initial_pose;
  Backend active_backend{Backend::kGnssImu};
  HandoffPhase phase{HandoffPhase::kGnssApproach};
  std::size_t switch_count{0U};
};

class LocalizationHandoff
{
public:
  explicit LocalizationHandoff(HandoffConfig config);

  HandoffUpdate observe(
    Backend backend, const nav_msgs::msg::Odometry & candidate,
    double receipt_time_sec);

  Backend active_backend() const noexcept;
  HandoffPhase phase() const noexcept;
  std::size_t switch_count() const noexcept;

private:
  struct SourceState
  {
    std::optional<nav_msgs::msg::Odometry> odometry;
    std::int64_t stamp_ns{-1};
    double receipt_time_sec{-1.0};
  };

  struct Correction
  {
    double x_m{0.0};
    double y_m{0.0};
    double yaw_rad{0.0};
    double start_time_sec{0.0};
    bool active{false};
  };

  HandoffUpdate current_update() const;
  bool validate_candidate(
    const nav_msgs::msg::Odometry & candidate, double receipt_time_sec,
    const SourceState & state) const;
  bool fresh(const SourceState & state, double now_sec) const;
  bool poses_agree(
    const nav_msgs::msg::Odometry & lhs,
    const nav_msgs::msg::Odometry & rhs) const;
  bool near(
    const nav_msgs::msg::Odometry & candidate, const Point2 & point,
    double radius_m) const;
  void begin_switch(
    Backend backend, HandoffPhase phase,
    const nav_msgs::msg::Odometry & first_candidate, double receipt_time_sec);
  std::optional<nav_msgs::msg::Odometry> make_canonical(
    const nav_msgs::msg::Odometry & candidate, double receipt_time_sec);
  SourceState & source(Backend backend);
  const SourceState & source(Backend backend) const;

  HandoffConfig config_;
  SourceState gnss_;
  SourceState fastlio_;
  Backend active_backend_{Backend::kGnssImu};
  HandoffPhase phase_{HandoffPhase::kGnssApproach};
  std::size_t switch_count_{0U};
  bool initial_pose_sent_{false};
  Correction correction_;
  std::optional<nav_msgs::msg::Odometry> last_canonical_;
  std::int64_t last_canonical_stamp_ns_{-1};
};

const char * to_string(Backend backend) noexcept;
const char * to_string(HandoffPhase phase) noexcept;

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__GNSS_SHADOW__LOCALIZATION_HANDOFF_HPP_
