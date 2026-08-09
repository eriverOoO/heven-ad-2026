#ifndef AD_PLANNER__DATA_LOADER_HPP_
#define AD_PLANNER__DATA_LOADER_HPP_

#include <filesystem>

#include "ad_planner/common/types.hpp"

namespace ad_planner {

class DataLoader {
public:
  static Route load_path(const std::filesystem::path &path);
};

} // namespace ad_planner

#endif // AD_PLANNER__DATA_LOADER_HPP_
