// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace fast_lio {

enum class Mode { kMapping, kLocalization };

enum class ScanTimingMode { kRolling, kInstantaneous };

inline Mode parse_mode(const std::string &value) {
  if (value == "mapping") return Mode::kMapping;
  if (value == "localization") return Mode::kLocalization;
  throw std::invalid_argument("mode must be mapping or localization");
}

inline ScanTimingMode parse_scan_timing_mode(const std::string &value) {
  if (value == "rolling") return ScanTimingMode::kRolling;
  if (value == "instantaneous") return ScanTimingMode::kInstantaneous;
  throw std::invalid_argument(
      "preprocess.scan_timing_mode must be rolling or instantaneous");
}

struct ScanTimingPolicy {
  static double effective_duration(ScanTimingMode mode,
                                   double measured_duration_sec,
                                   double mean_duration_sec) {
    if (!std::isfinite(measured_duration_sec) ||
        !std::isfinite(mean_duration_sec) || measured_duration_sec < 0.0 ||
        mean_duration_sec < 0.0) {
      throw std::invalid_argument("scan durations must be finite and nonnegative");
    }
    if (mode == ScanTimingMode::kInstantaneous) return 0.0;
    if (measured_duration_sec < 0.5 * mean_duration_sec) {
      return mean_duration_sec;
    }
    return measured_duration_sec;
  }

  static constexpr bool applies_point_undistortion(ScanTimingMode mode) {
    return mode == ScanTimingMode::kRolling;
  }
};

struct ModePolicy {
  static constexpr bool requires_map(Mode mode) {
    return mode == Mode::kLocalization;
  }
  static constexpr bool may_mutate_map(Mode mode) {
    return mode == Mode::kMapping;
  }
  static constexpr bool may_save_map(Mode mode) {
    return mode == Mode::kMapping;
  }
  static constexpr bool accepts_sensor_data(Mode mode, bool initial_pose_ready,
                                            bool map_ready) {
    return initial_pose_ready && (!requires_map(mode) || map_ready);
  }
  static constexpr bool accepts_initial_pose_update(
      Mode mode, bool initial_pose_ready) {
    return !initial_pose_ready || mode == Mode::kLocalization;
  }
};

}  // namespace fast_lio
