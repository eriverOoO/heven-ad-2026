// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace fast_lio {

enum class LocalizationHealthState {
  kInitializing,
  kHealthy,
  kLost,
};

struct LocalizationHealthConfig {
  std::size_t minimum_effective_points;
  double maximum_mean_residual_m;
  std::size_t healthy_scan_count;
  std::size_t lost_scan_count;
};

class LocalizationHealth {
 public:
  explicit LocalizationHealth(LocalizationHealthConfig config) : config_(config) {
    if (config_.minimum_effective_points == 0 ||
        !std::isfinite(config_.maximum_mean_residual_m) ||
        config_.maximum_mean_residual_m <= 0.0 ||
        config_.healthy_scan_count == 0 || config_.lost_scan_count == 0) {
      throw std::invalid_argument("invalid fixed-map localization health configuration");
    }
  }

  bool observe(std::size_t effective_points, double mean_residual_m) {
    const bool valid = effective_points >= config_.minimum_effective_points &&
                       std::isfinite(mean_residual_m) && mean_residual_m >= 0.0 &&
                       mean_residual_m <= config_.maximum_mean_residual_m;
    if (valid) {
      ++healthy_scans_;
      unhealthy_scans_ = 0;
      if (healthy_scans_ >= config_.healthy_scan_count) {
        state_ = LocalizationHealthState::kHealthy;
      }
    } else {
      healthy_scans_ = 0;
      ++unhealthy_scans_;
      if (unhealthy_scans_ >= config_.lost_scan_count) {
        state_ = LocalizationHealthState::kLost;
      }
    }
    return state_ == LocalizationHealthState::kHealthy;
  }

  LocalizationHealthState state() const { return state_; }

  void mark_stale() {
    state_ = LocalizationHealthState::kLost;
    healthy_scans_ = 0;
    unhealthy_scans_ = config_.lost_scan_count;
  }

 private:
  LocalizationHealthConfig config_;
  LocalizationHealthState state_{LocalizationHealthState::kInitializing};
  std::size_t healthy_scans_{0};
  std::size_t unhealthy_scans_{0};
};

inline const char *to_string(LocalizationHealthState state) {
  switch (state) {
    case LocalizationHealthState::kInitializing:
      return "initializing";
    case LocalizationHealthState::kHealthy:
      return "healthy";
    case LocalizationHealthState::kLost:
      return "lost";
  }
  return "unknown";
}

}  // namespace fast_lio
