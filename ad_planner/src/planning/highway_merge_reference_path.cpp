#include "ad_planner/planning/highway_merge_reference_path.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ad_planner {
namespace {

bool finite(double value) { return std::isfinite(value); }

double heading_of(const ad_control::Point3 &from, const ad_control::Point3 &to) {
  return std::atan2(to.y - from.y, to.x - from.x);
}

double gap_of(const ad_control::Point3 &a, const ad_control::Point3 &b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

}  // namespace

HighwayMergeReferencePath build_highway_merge_reference_path(
    const ReferenceLane &source_lane, const ReferenceLane &target_lane,
    const double merge_complete_route_s_m,
    const HighwayMergeReferencePathConfig &config) {
  if (source_lane.points.size() < 2U || target_lane.points.size() < 2U) {
    throw std::invalid_argument(
        "highway merge reference path: source / target lane has too few points");
  }
  if (!finite(merge_complete_route_s_m) ||
      !finite(config.target_continuation_m) ||
      config.target_continuation_m <= 0.0) {
    throw std::invalid_argument(
        "highway merge reference path: invalid merge-complete station or "
        "continuation length");
  }

  const double source_first_s = source_lane.points.front().route_s_m;
  const double source_last_s = source_lane.points.back().route_s_m;
  if (!finite(source_first_s) || !finite(source_last_s) ||
      !(source_last_s > source_first_s)) {
    throw std::invalid_argument(
        "highway merge reference path: source lane stations are not increasing");
  }
  if (std::abs(source_last_s - merge_complete_route_s_m) >
      config.station_consistency_margin_m) {
    throw std::invalid_argument(
        "highway merge reference path: source lane does not end at the "
        "merge-complete station");
  }

  HighwayMergeReferencePath result;
  result.splice_route_s_m = merge_complete_route_s_m;
  result.source_first_route_s_m = source_first_s;

  // SOURCE section: the real acceleration-lane centerline, verbatim.
  double previous_station = -std::numeric_limits<double>::infinity();
  for (const auto &point : source_lane.points) {
    if (!finite(point.pose.x) || !finite(point.pose.y) ||
        !finite(point.route_s_m)) {
      throw std::invalid_argument(
          "highway merge reference path: non-finite source lane point");
    }
    if (!(point.route_s_m >= previous_station - 1e-6)) {
      throw std::invalid_argument(
          "highway merge reference path: source lane station regressed");
    }
    previous_station = point.route_s_m;
    result.route.points.push_back(
        ad_control::Point3{point.pose.x, point.pose.y, 0.0});
  }
  result.source_point_count = result.route.points.size();

  // TARGET section: real route:0 mainline, strictly past the merge-complete
  // coincidence, so the join point is contributed once (by the source section).
  const double continuation_end_s =
      merge_complete_route_s_m + config.target_continuation_m;
  previous_station = merge_complete_route_s_m;
  for (const auto &point : target_lane.points) {
    if (!(point.route_s_m > merge_complete_route_s_m + 1e-6)) {
      continue;
    }
    if (point.route_s_m > continuation_end_s + 1e-6) {
      break;
    }
    if (!finite(point.pose.x) || !finite(point.pose.y) ||
        !finite(point.route_s_m)) {
      throw std::invalid_argument(
          "highway merge reference path: non-finite target lane point");
    }
    if (!(point.route_s_m >= previous_station - 1e-6)) {
      throw std::invalid_argument(
          "highway merge reference path: target lane station regressed");
    }
    previous_station = point.route_s_m;
    result.route.points.push_back(
        ad_control::Point3{point.pose.x, point.pose.y, 0.0});
    result.target_last_route_s_m = point.route_s_m;
  }
  result.target_point_count =
      result.route.points.size() - result.source_point_count;
  if (result.target_point_count < 2U) {
    throw std::invalid_argument(
        "highway merge reference path: target lane does not extend far enough "
        "past the merge-complete station");
  }

  // Join contract: the source last point and the first target point.
  const auto &join_a = result.route.points[result.source_point_count - 1U];
  const auto &join_b = result.route.points[result.source_point_count];
  result.splice_join_gap_m = gap_of(join_a, join_b);
  const auto &before_a = result.route.points[result.source_point_count - 2U];
  const auto &after_b = result.route.points[result.source_point_count + 1U];
  result.splice_join_heading_delta_rad = std::abs(std::remainder(
      heading_of(join_a, join_b) - heading_of(before_a, join_a),
      2.0 * M_PI));
  const double post_join_heading_delta = std::abs(std::remainder(
      heading_of(join_b, after_b) - heading_of(join_a, join_b), 2.0 * M_PI));
  if (result.splice_join_gap_m > config.maximum_join_gap_m ||
      result.splice_join_heading_delta_rad >
          config.maximum_join_heading_delta_rad ||
      post_join_heading_delta > config.maximum_join_heading_delta_rad) {
    throw std::invalid_argument(
        "highway merge reference path: source -> target join is not "
        "geometrically continuous");
  }

  // No consecutive duplicate points (the tracker's segment math needs distinct
  // points); bounded spacing everywhere.
  for (std::size_t i = 1U; i < result.route.points.size(); ++i) {
    const double spacing =
        gap_of(result.route.points[i - 1U], result.route.points[i]);
    if (!(spacing > 1e-6)) {
      throw std::invalid_argument(
          "highway merge reference path: duplicate consecutive points");
    }
  }

  result.route.closed = false;
  return result;
}

}  // namespace ad_planner
