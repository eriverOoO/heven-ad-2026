#ifndef AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_FACTORY_HPP_
#define AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_FACTORY_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ad_control/common/types.hpp"
#include "ad_control/lateral/path_tracking_controller.hpp"

namespace ad_control {

enum class PathTrackingBackend {
  kStanley,
  kProfileStanley,
};

class PathTrackingParameterProvider {
public:
  virtual ~PathTrackingParameterProvider() = default;
  virtual double get_double(const std::string &name, double default_value) = 0;
  virtual int get_int(const std::string &name, int default_value) = 0;
  virtual std::vector<double>
  get_double_array(const std::string &,
                   const std::vector<double> &default_value) {
    return default_value;
  }
};

PathTrackingBackend parse_path_tracking_backend(const std::string &backend);

std::unique_ptr<PathTrackingController>
make_path_tracking_controller(PathTrackingBackend backend, Route route,
                              PathTrackingParameterProvider &parameters);

std::unique_ptr<PathTrackingController>
make_path_tracking_controller(const std::string &backend, Route route,
                              PathTrackingParameterProvider &parameters);

} // namespace ad_control

#endif // AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_FACTORY_HPP_
