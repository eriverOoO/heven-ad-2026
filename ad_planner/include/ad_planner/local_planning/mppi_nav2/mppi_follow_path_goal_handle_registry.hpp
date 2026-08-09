#ifndef AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_GOAL_HANDLE_REGISTRY_HPP_
#define AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_GOAL_HANDLE_REGISTRY_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace ad_planner
{

template<typename HandleT>
class MppiFollowPathGoalHandleRegistry
{
public:
  void record_accepted_response(
    const std::uint64_t generation,
    HandleT handle,
    const bool goal_response_was_pending,
    const bool cancel_requested,
    const std::optional<std::uint64_t> accepted_generation)
  {
    if (!goal_response_was_pending) {
      return;
    }

    if (cancel_requested) {
      handles_[generation] = std::move(handle);
      return;
    }

    if (!accepted_generation || *accepted_generation != generation) {
      return;
    }

    if (current_generation_ && *current_generation_ != generation) {
      handles_.erase(*current_generation_);
    }
    current_generation_ = generation;
    handles_[generation] = std::move(handle);
  }

  std::optional<HandleT> take_for_cancel(const std::uint64_t generation)
  {
    const auto handle = handles_.find(generation);
    if (handle == handles_.end()) {
      return std::nullopt;
    }

    std::optional<HandleT> result(std::move(handle->second));
    handles_.erase(handle);
    if (current_generation_ && *current_generation_ == generation) {
      current_generation_.reset();
    }
    return result;
  }

  void erase_result(const std::uint64_t generation)
  {
    handles_.erase(generation);
    if (current_generation_ && *current_generation_ == generation) {
      current_generation_.reset();
    }
  }

  bool contains(const std::uint64_t generation) const
  {
    return handles_.find(generation) != handles_.end();
  }

  std::size_t size() const noexcept
  {
    return handles_.size();
  }

  std::optional<std::uint64_t> current_generation() const noexcept
  {
    return current_generation_;
  }

private:
  std::optional<std::uint64_t> current_generation_;
  std::unordered_map<std::uint64_t, HandleT> handles_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__MPPI_FOLLOW_PATH_GOAL_HANDLE_REGISTRY_HPP_
