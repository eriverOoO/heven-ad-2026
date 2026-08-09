#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FACTORY_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FACTORY_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

enum class LocalMotionBackendKind
{
  kDwa,
  kFrenetLattice,
  kMppiNav2,
};

class LocalMotionParameterProvider
{
public:
  virtual ~LocalMotionParameterProvider() = default;

  virtual double get_double(const std::string & name, double default_value) = 0;
  virtual int get_int(const std::string & name, int default_value) = 0;
  virtual std::vector<double> get_double_array(
    const std::string &, const std::vector<double> & default_value)
  {
    return default_value;
  }
};

LocalMotionBackendKind parse_local_motion_backend(const std::string & backend);

std::unique_ptr<LocalMotionBackend> make_local_motion_backend(
  LocalMotionBackendKind backend, LocalMotionParameterProvider & parameters);

std::unique_ptr<LocalMotionBackend> make_local_motion_backend(
  const std::string & backend, LocalMotionParameterProvider & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FACTORY_HPP_
