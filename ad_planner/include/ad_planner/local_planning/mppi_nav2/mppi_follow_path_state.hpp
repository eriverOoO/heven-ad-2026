#ifndef AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_STATE_HPP_
#define AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_STATE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <nav_msgs/msg/path.hpp>

namespace ad_planner
{

struct MppiFollowPathPoint
{
  double position_x{0.0};
  double position_y{0.0};
  double position_z{0.0};
  double orientation_x{0.0};
  double orientation_y{0.0};
  double orientation_z{0.0};
  double orientation_w{0.0};
};

bool operator==(
  const MppiFollowPathPoint & lhs,
  const MppiFollowPathPoint & rhs) noexcept;
bool operator!=(
  const MppiFollowPathPoint & lhs,
  const MppiFollowPathPoint & rhs) noexcept;

struct MppiFollowPathCanonicalPath
{
  std::string frame_id;
  std::vector<MppiFollowPathPoint> points;
};

bool operator==(
  const MppiFollowPathCanonicalPath & lhs,
  const MppiFollowPathCanonicalPath & rhs) noexcept;
bool operator!=(
  const MppiFollowPathCanonicalPath & lhs,
  const MppiFollowPathCanonicalPath & rhs) noexcept;

struct MppiFollowPathAdmission
{
  bool valid{false};
  std::string reason{"invalid"};
  MppiFollowPathCanonicalPath path;
};

MppiFollowPathAdmission canonicalize_mppi_follow_path(
  const nav_msgs::msg::Path & path, std::size_t maximum_pose_count);

enum class MppiFollowPathActionType
{
  kSendGoal,
  kCancelGoal,
};

struct MppiFollowPathAction
{
  MppiFollowPathActionType type{MppiFollowPathActionType::kSendGoal};
  std::uint64_t generation{0U};
};

struct MppiFollowPathEffects
{
  std::optional<std::uint64_t> new_path_generation;
  std::vector<MppiFollowPathAction> actions;
  bool goal_response_was_pending{false};
};

enum class MppiFollowPathResultCode
{
  kSucceeded,
  kAborted,
  kCanceled,
};

struct MppiFollowPathStateConfig
{
  std::int64_t path_timeout_ns{500'000'000};
};

struct MppiFollowPathStateSnapshot
{
  bool has_path{false};
  std::optional<std::uint64_t> current_generation;
  std::optional<std::uint64_t> accepted_generation;
  std::size_t pending_goal_response_count{0U};
  std::int64_t receipt_steady_ns{0};
};

class MppiFollowPathState
{
public:
  explicit MppiFollowPathState(MppiFollowPathStateConfig config);

  MppiFollowPathEffects observe_path(
    const MppiFollowPathCanonicalPath & path,
    std::int64_t receipt_steady_ns);
  MppiFollowPathEffects poll(
    std::int64_t now_steady_ns, bool action_server_ready);
  MppiFollowPathEffects on_goal_response(
    std::uint64_t generation, bool accepted);
  MppiFollowPathEffects on_goal_result(
    std::uint64_t generation, MppiFollowPathResultCode result);
  MppiFollowPathEffects deactivate();

  MppiFollowPathStateSnapshot snapshot() const;

private:
  void clear_current_without_cancel() noexcept;

  MppiFollowPathStateConfig config_;
  std::uint64_t next_generation_{0U};
  std::optional<MppiFollowPathCanonicalPath> current_path_;
  std::int64_t receipt_steady_ns_{0};
  std::optional<std::uint64_t> current_generation_;
  std::optional<std::uint64_t> waiting_generation_;
  std::optional<std::uint64_t> accepted_generation_;
  std::unordered_set<std::uint64_t> pending_goal_responses_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_STATE_HPP_
