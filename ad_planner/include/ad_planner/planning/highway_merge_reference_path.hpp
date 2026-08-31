#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_REFERENCE_PATH_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_REFERENCE_PATH_HPP_

#include <cstddef>

#include "ad_control/common/types.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner {

// Configuration for the online highway-merge lateral reference path. Only the
// forward continuation length is tunable; no lateral-shape parameter exists
// because the reference is the real map geometry (see build_...).
struct HighwayMergeReferencePathConfig {
  // How much real route:0 mainline geometry to append after the merge-complete
  // splice, so the path tracker never sees a path-end artifact while the ego is
  // still finishing the merge / just past it.
  double target_continuation_m{200.0};
  // The source-lane last station and the declared merge-complete station must
  // agree within this margin (mirrors ad_highway_merge_gap_risk's own check).
  double station_consistency_margin_m{2.0};
  // Reject the build if the source->target join is not geometrically
  // continuous (the real acceleration lane tapers fully into route:0, so this
  // is ~0 in the shipped corridor).
  double maximum_join_gap_m{1.0};
  double maximum_join_heading_delta_rad{0.20};
};

// The built merge reference: the real route:0:left:1 acceleration-lane
// centerline followed by the real route:0 mainline centerline past the merge
// point. Map frame, x/y only (ReferencePoint carries no z, so Route z is 0).
struct HighwayMergeReferencePath {
  ad_control::Route route;
  std::size_t source_point_count{0};   // route:0:left:1 points
  std::size_t target_point_count{0};   // route:0 continuation points
  double splice_route_s_m{0.0};        // == merge-complete station
  double splice_join_gap_m{0.0};       // Euclidean gap across the splice
  double splice_join_heading_delta_rad{0.0};
  double source_first_route_s_m{0.0};
  double target_last_route_s_m{0.0};
};

// Build the source-grounded route:0:left:1 -> route:0 merge reference path.
//
// SOURCE section: every point of `source_lane` (route:0:left:1), whose own
// geometry already tapers from ~3.94 m off route:0 to 0.0 m -- no synthetic
// lane-change curve is added.
// TARGET section: `target_lane` (route:0) points with route_s strictly greater
// than `merge_complete_route_s_m`, up to `+ target_continuation_m`. The
// coincident merge-complete point is contributed once, by the source section.
//
// Throws std::invalid_argument on any inconsistency (too few points, non-finite
// coordinate, non-monotonic station, station mismatch, discontinuous join,
// insufficient target continuation). Callers treat a throw as "feature inert".
HighwayMergeReferencePath build_highway_merge_reference_path(
    const ReferenceLane &source_lane, const ReferenceLane &target_lane,
    double merge_complete_route_s_m,
    const HighwayMergeReferencePathConfig &config);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__HIGHWAY_MERGE_REFERENCE_PATH_HPP_
