#ifndef AD_PLANNER__COMMON__PARAMETER_VALIDATION_HPP_
#define AD_PLANNER__COMMON__PARAMETER_VALIDATION_HPP_

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ad_planner
{

inline double positive_finite_parameter(double value, const std::string & name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(name + " must be finite and positive");
  }
  return value;
}

inline std::size_t positive_size_parameter(int value, const std::string & name)
{
  if (value <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace ad_planner

#endif  // AD_PLANNER__COMMON__PARAMETER_VALIDATION_HPP_
