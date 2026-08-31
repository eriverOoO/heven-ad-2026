#ifndef AD_PLANNER__PLANNING__EXTERNAL_SPEED_LIMIT_HPP_
#define AD_PLANNER__PLANNING__EXTERNAL_SPEED_LIMIT_HPP_

#include <initializer_list>
#include <optional>

namespace ad_planner {

// Compose two independent external longitudinal speed limits (each already an
// upper bound, or std::nullopt for "this source imposes no limit this tick").
//
// External longitudinal constraints are mathematical upper bounds, not ordered
// commands: the result is the smallest present value, or std::nullopt when both
// are absent. combine_speed_limits(a, b) == combine_speed_limits(b, a) for every
// input, so the planner never depends on the order it consults its constraint
// sources. NaN inputs are ignored (treated as "no limit") so a malformed source
// can never poison the combined bound.
std::optional<double> combine_speed_limits(std::optional<double> a,
                                           std::optional<double> b);

// Fold an arbitrary set of external longitudinal speed limits with the binary
// combiner above. min is associative and combine_speed_limits is symmetric, so
// the result is independent of the order the sources appear in the list; an
// empty list or a list of all-absent / all-non-finite entries yields
// std::nullopt. Used where more than two independent sources (cut-in,
// roundabout, highway merge) are combined.
std::optional<double>
combine_speed_limits(std::initializer_list<std::optional<double>> limits);

} // namespace ad_planner

#endif // AD_PLANNER__PLANNING__EXTERNAL_SPEED_LIMIT_HPP_
