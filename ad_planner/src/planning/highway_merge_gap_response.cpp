#include "ad_planner/planning/highway_merge_gap_response.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ad_planner
{
namespace
{

void require_nonnegative(const double value, const char * const name)
{
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
  }
}

void require_positive(const double value, const char * const name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}

// Reason precedence, most limiting first. Higher value wins arbitration.
enum class ViolationKind : std::uint8_t
{
  kFrontGap = 1U,
  kRearGap = 2U,
  kRearClosing = 3U,
  kPredictedRouteConflict = 4U,
  kInsufficientPrediction = 5U,
  kAlongside = 6U,
};

struct Violation
{
  const HighwayMergeGapResponseObjectInput * object{nullptr};
  ViolationKind kind{ViolationKind::kFrontGap};
  double metric{0.0};  // smaller = more restrictive within a kind
};

HighwayMergeGapResponseReason reason_of(const ViolationKind kind)
{
  switch (kind) {
    case ViolationKind::kAlongside:
      return HighwayMergeGapResponseReason::kAlongside;
    case ViolationKind::kInsufficientPrediction:
      return HighwayMergeGapResponseReason::kInsufficientPrediction;
    case ViolationKind::kPredictedRouteConflict:
      return HighwayMergeGapResponseReason::kPredictedRouteConflict;
    case ViolationKind::kRearClosing:
      return HighwayMergeGapResponseReason::kRearClosing;
    case ViolationKind::kRearGap:
      return HighwayMergeGapResponseReason::kRearGap;
    case ViolationKind::kFrontGap:
    default:
      return HighwayMergeGapResponseReason::kFrontGap;
  }
}

// Strict-weak "ranked below" order for std::max_element: the returned element is
// the most limiting -- highest reason precedence, then smallest metric, then
// lexicographically smallest UUID.
bool ranked_below(const Violation & lhs, const Violation & rhs)
{
  if (lhs.kind != rhs.kind) {
    return static_cast<unsigned>(lhs.kind) < static_cast<unsigned>(rhs.kind);
  }
  if (lhs.metric != rhs.metric) {
    return lhs.metric > rhs.metric;
  }
  return rhs.object->object_id < lhs.object->object_id;
}

}  // namespace

HighwayMergeGapResponseParameters HighwayMergeGapResponseParameters::validated() const
{
  require_positive(minimum_front_time_headway_s, "minimum_front_time_headway_s");
  require_positive(minimum_rear_time_headway_s, "minimum_rear_time_headway_s");
  require_positive(minimum_rear_closing_time_s, "minimum_rear_closing_time_s");
  require_positive(minimum_predicted_route_gap_m, "minimum_predicted_route_gap_m");
  require_nonnegative(merge_standoff_m, "merge_standoff_m");
  require_positive(comfortable_deceleration_mps2, "comfortable_deceleration_mps2");
  require_nonnegative(stopped_speed_threshold_mps, "stopped_speed_threshold_mps");
  require_positive(speed_epsilon_mps, "speed_epsilon_mps");
  if (maximum_relevant_objects == 0U) {
    throw std::invalid_argument("maximum_relevant_objects must be positive");
  }
  return *this;
}

HighwayMergeGapResponseResult compute_highway_merge_gap_response(
  const HighwayMergeGapResponseInput & input,
  const HighwayMergeGapResponseParameters & parameters)
{
  const auto config = parameters.validated();
  const auto finite = [](const double value) {return std::isfinite(value);};
  if (!finite(input.ego_merge_time_s) || !finite(input.ego_speed_mps) ||
    !finite(input.ego_route_distance_to_merge_m))
  {
    throw std::invalid_argument("ego response facts must be finite");
  }
  if (input.ego_speed_mps < 0.0) {
    throw std::invalid_argument("ego speed must be nonnegative");
  }
  if (input.ego_merge_timing_valid && input.ego_merge_time_s < 0.0) {
    throw std::invalid_argument("valid ego merge time must be nonnegative");
  }

  HighwayMergeGapResponseResult result;
  result.ego_speed_mps = std::max(0.0, input.ego_speed_mps);
  result.ego_route_distance_to_merge_m = input.ego_route_distance_to_merge_m;
  result.ego_merge_timing_valid = input.ego_merge_timing_valid;
  result.ego_merge_time_s = input.ego_merge_timing_valid ? input.ego_merge_time_s : 0.0;
  result.available_distance_m =
    std::max(0.0, input.ego_route_distance_to_merge_m - config.merge_standoff_m);
  result.comfortable_stop_distance_m = result.ego_speed_mps * result.ego_speed_mps /
    (2.0 * config.comfortable_deceleration_mps2);

  const std::size_t admitted = std::min(
    input.relevant_objects.size(), config.maximum_relevant_objects);
  result.relevant_object_count = admitted;
  result.rejected_over_budget =
    (input.relevant_objects.size() - admitted) + input.relevant_objects_omitted;

  // Not applicable: ego far before the zone or past the merge reference. A
  // non-restrictive inactive value; a consumer must check `active`.
  if (!input.applicable) {
    return result;
  }
  result.active = true;
  result.complete_prediction_coverage = true;

  const bool stopped = result.ego_speed_mps < config.stopped_speed_threshold_mps;

  // Nearest object ahead / behind the merge reference at the ego merge time.
  const HighwayMergeGapResponseObjectInput * front = nullptr;
  const HighwayMergeGapResponseObjectInput * rear = nullptr;
  const HighwayMergeGapResponseObjectInput * rear_closing = nullptr;
  double rear_closing_metric = 0.0;
  bool have_predicted_clearance = false;
  double min_predicted_gap = 0.0;
  std::vector<Violation> violations;
  violations.reserve(admitted);

  const auto headway = [&config](const double gap, const double speed) {
      return gap / std::max(speed, config.speed_epsilon_mps);
    };

  for (std::size_t index = 0U; index < admitted; ++index) {
    const auto & object = input.relevant_objects[index];
    if (!finite(object.delta_s_now_m) || !finite(object.object_longitudinal_speed_mps) ||
      !finite(object.delta_s_at_merge_m) || !finite(object.longitudinal_closing_speed_mps) ||
      !finite(object.time_to_route_coincidence_s) || !finite(object.predicted_min_route_gap_m))
    {
      throw std::invalid_argument("relevant merge object facts must be finite");
    }
    const int relation_flags = (object.is_ahead_at_merge ? 1 : 0) +
      (object.is_behind_at_merge ? 1 : 0) + (object.is_alongside_at_merge ? 1 : 0);
    if (relation_flags > 1) {
      throw std::invalid_argument("merge relation flags are not mutually exclusive");
    }
    if (relation_flags == 1 && !object.delta_s_at_merge_valid) {
      throw std::invalid_argument("merge relation flag set without a valid delta_s_at_merge");
    }
    if (object.predicted_min_route_gap_valid && object.predicted_min_route_gap_m < 0.0) {
      throw std::invalid_argument("predicted min route gap must be nonnegative");
    }

    result.complete_prediction_coverage =
      result.complete_prediction_coverage && object.prediction_covers_merge_time;

    if (object.predicted_min_route_gap_valid) {
      if (!have_predicted_clearance ||
        object.predicted_min_route_gap_m < min_predicted_gap)
      {
        have_predicted_clearance = true;
        min_predicted_gap = object.predicted_min_route_gap_m;
      }
    }

    if (object.delta_s_at_merge_valid && object.is_ahead_at_merge) {
      if (front == nullptr || object.delta_s_at_merge_m < front->delta_s_at_merge_m ||
        (object.delta_s_at_merge_m == front->delta_s_at_merge_m &&
        object.object_id < front->object_id))
      {
        front = &object;
      }
    }
    if (object.delta_s_at_merge_valid && object.is_behind_at_merge) {
      // "behind" delta is negative; closest behind is the largest (least
      // negative) delta_s_at_merge_m.
      if (rear == nullptr || object.delta_s_at_merge_m > rear->delta_s_at_merge_m ||
        (object.delta_s_at_merge_m == rear->delta_s_at_merge_m &&
        object.object_id < rear->object_id))
      {
        rear = &object;
      }
    }

    // Rear-closing is a current-time fact: an object currently behind the ego
    // (delta_s_now_m < 0) whose route-station gap to the ego is shrinking.
    // longitudinal_gap_closing is symmetric, so it is gated on delta_s_now_m to
    // avoid tripping on a slow front vehicle the ego is overtaking.
    bool closing_violation = false;
    double closing_metric = 0.0;
    if (object.delta_s_now_m < 0.0 && object.longitudinal_gap_closing) {
      closing_metric = (object.time_to_route_coincidence_valid &&
        object.time_to_route_coincidence_s >= 0.0) ?
        object.time_to_route_coincidence_s : 0.0;
      if (rear_closing == nullptr || closing_metric < rear_closing_metric) {
        rear_closing = &object;
        rear_closing_metric = closing_metric;
      }
      if (!object.time_to_route_coincidence_valid ||
        object.time_to_route_coincidence_s < config.minimum_rear_closing_time_s)
      {
        closing_violation = true;
      }
    }

    // Per-object violation classification (worst applicable), reason precedence
    // order.
    if (object.is_alongside_at_merge) {
      violations.push_back(
        Violation{&object, ViolationKind::kAlongside, std::fabs(object.delta_s_at_merge_m)});
      continue;
    }
    if (!object.delta_s_at_merge_valid || !object.prediction_covers_merge_time ||
      !object.predicted_min_route_gap_valid)
    {
      violations.push_back(
        Violation{&object, ViolationKind::kInsufficientPrediction, 0.0});
      continue;
    }
    if (object.predicted_min_route_gap_m < config.minimum_predicted_route_gap_m) {
      violations.push_back(
        Violation{
          &object, ViolationKind::kPredictedRouteConflict, object.predicted_min_route_gap_m});
      continue;
    }
    if (closing_violation) {
      violations.push_back(
        Violation{&object, ViolationKind::kRearClosing, closing_metric});
      continue;
    }
    if (object.is_behind_at_merge) {
      const double rear_gap = -object.delta_s_at_merge_m;
      const double rear_headway = headway(rear_gap, object.object_longitudinal_speed_mps);
      if (rear_headway < config.minimum_rear_time_headway_s) {
        violations.push_back(Violation{&object, ViolationKind::kRearGap, rear_headway});
        continue;
      }
    }
    if (object.is_ahead_at_merge) {
      const double front_gap = object.delta_s_at_merge_m;
      const double front_headway = headway(front_gap, result.ego_speed_mps);
      if (front_headway < config.minimum_front_time_headway_s) {
        violations.push_back(Violation{&object, ViolationKind::kFrontGap, front_headway});
        continue;
      }
    }
  }

  if (front != nullptr) {
    result.front_object_valid = true;
    result.front_object_id = front->object_id;
    result.front_gap_m = front->delta_s_at_merge_m;
    result.front_time_headway_valid = true;
    result.front_time_headway_s = headway(front->delta_s_at_merge_m, result.ego_speed_mps);
  }
  if (rear != nullptr) {
    result.rear_object_valid = true;
    result.rear_object_id = rear->object_id;
    result.rear_gap_m = -rear->delta_s_at_merge_m;
    result.rear_time_headway_valid = true;
    result.rear_time_headway_s =
      headway(-rear->delta_s_at_merge_m, rear->object_longitudinal_speed_mps);
  }
  if (rear_closing != nullptr) {
    result.rear_closing_object_valid = true;
    result.rear_closing_object_id = rear_closing->object_id;
    result.rear_closing_time_valid = rear_closing->time_to_route_coincidence_valid;
    result.rear_closing_time_s = rear_closing->time_to_route_coincidence_valid ?
      rear_closing->time_to_route_coincidence_s : 0.0;
  }
  if (have_predicted_clearance) {
    result.predicted_route_clearance_valid = true;
    result.minimum_predicted_route_gap_m = min_predicted_gap;
  }

  const bool ego_ok = input.ego_merge_timing_valid && !stopped;

  if (ego_ok && result.rejected_over_budget == 0U && violations.empty()) {
    result.action = HighwayMergeGapResponseAction::kMergeReady;
    result.reason = HighwayMergeGapResponseReason::kClearGap;
    return result;
  }

  if (!violations.empty()) {
    const auto limiting = *std::max_element(
      violations.begin(), violations.end(),
      [](const Violation & lhs, const Violation & rhs) {return ranked_below(lhs, rhs);});
    result.has_source = true;
    result.source_object_id = limiting.object->object_id;
    result.reason = reason_of(limiting.kind);
  } else {
    // No blocking object, but still not ready: invalid ego timing / stopped, or
    // a relevant object was dropped over budget.
    result.reason = (!input.ego_merge_timing_valid || stopped) ?
      HighwayMergeGapResponseReason::kInvalidEgoState :
      HighwayMergeGapResponseReason::kInsufficientPrediction;
    result.complete_prediction_coverage = false;
  }

  // A stopped ego holds. Otherwise the strict '>' boundary preserves the full
  // comfortable stopping margin before the merge decision boundary.
  result.action = (!stopped &&
    result.available_distance_m > result.comfortable_stop_distance_m) ?
    HighwayMergeGapResponseAction::kWait : HighwayMergeGapResponseAction::kHold;
  return result;
}

}  // namespace ad_planner
