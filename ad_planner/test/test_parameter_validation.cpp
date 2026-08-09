#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_planner/common/parameter_validation.hpp"

namespace
{

using ad_planner::positive_finite_parameter;
using ad_planner::positive_size_parameter;

TEST(PlannerParameterValidation, RejectsNonFiniteOrNonPositiveControlPeriod)
{
  EXPECT_DOUBLE_EQ(positive_finite_parameter(0.05, "control_period_sec"), 0.05);
  EXPECT_THROW(positive_finite_parameter(0.0, "control_period_sec"), std::invalid_argument);
  EXPECT_THROW(positive_finite_parameter(-0.1, "control_period_sec"), std::invalid_argument);
  EXPECT_THROW(
    positive_finite_parameter(
      std::numeric_limits<double>::infinity(), "control_period_sec"),
    std::invalid_argument);
}

TEST(PlannerParameterValidation, RejectsNonPositiveCountsBeforeUnsignedConversion)
{
  EXPECT_EQ(positive_size_parameter(1, "forward_window"), 1U);
  EXPECT_THROW(positive_size_parameter(0, "forward_window"), std::invalid_argument);
  EXPECT_THROW(positive_size_parameter(-1, "maximum_laps"), std::invalid_argument);
}

}  // namespace
