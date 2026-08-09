#ifndef AD_PLANNER__LOCAL_PLANNING__FRENET_GEOMETRY_HPP_
#define AD_PLANNER__LOCAL_PLANNING__FRENET_GEOMETRY_HPP_

#include <array>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct FrenetState
{
  double s_m{0.0};
  double s_dot_mps{0.0};
  double s_ddot_mps2{0.0};
  double d_m{0.0};
  double d_dot_mps{0.0};
  double d_ddot_mps2{0.0};
};

class QuinticPolynomial
{
public:
  QuinticPolynomial(
    double p0, double v0, double a0,
    double p1, double v1, double a1, double duration_s);

  double position(double t_s) const;
  double velocity(double t_s) const;
  double acceleration(double t_s) const;
  double jerk(double t_s) const;

private:
  std::array<long double, 6> coefficients_{};
};

class QuarticPolynomial
{
public:
  QuarticPolynomial(
    double p0, double v0, double a0,
    double v1, double a1, double duration_s);

  double position(double t_s) const;
  double velocity(double t_s) const;
  double acceleration(double t_s) const;
  double jerk(double t_s) const;

private:
  std::array<long double, 5> coefficients_{};
};

FrenetState project_to_frenet(const ReferenceLane & lane, const EgoState & ego);

TimedTrajectoryPoint frenet_to_cartesian(
  const ReferenceLane & lane, const FrenetState & state, double time_s);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__FRENET_GEOMETRY_HPP_
