#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>

namespace fast_lio {

struct TimedWheelVelocity {
  double stamp_sec{0.0};
  double forward_speed_mps{0.0};
  double forward_variance{0.0};
  std::optional<double> lateral_speed_mps;
  double lateral_variance{0.0};
  std::optional<double> distance_stamp_sec;
  std::optional<double> cumulative_forward_distance_m;
  std::optional<double> cumulative_lateral_distance_m;
};

class WheelVelocityBuffer {
 public:
  explicit WheelVelocityBuffer(
      std::size_t capacity,
      double maximum_distance_gap_sec = 0.5)
      : capacity_(capacity),
        maximum_distance_gap_sec_(maximum_distance_gap_sec) {
    if (capacity_ == 0) throw std::invalid_argument("wheel buffer capacity must be positive");
    if (!std::isfinite(maximum_distance_gap_sec_) ||
        maximum_distance_gap_sec_ <= 0.0) {
      throw std::invalid_argument(
          "maximum wheel distance gap must be finite and positive");
    }
  }

  bool push(TimedWheelVelocity sample) {
    if (!std::isfinite(sample.stamp_sec) || sample.stamp_sec <= 0.0 ||
        !std::isfinite(sample.forward_speed_mps) ||
        !std::isfinite(sample.forward_variance) || sample.forward_variance <= 0.0 ||
        (sample.lateral_speed_mps &&
         (!std::isfinite(*sample.lateral_speed_mps) ||
          !std::isfinite(sample.lateral_variance) ||
          sample.lateral_variance <= 0.0)) ||
        (sample.distance_stamp_sec &&
         (!std::isfinite(*sample.distance_stamp_sec) ||
          *sample.distance_stamp_sec <= 0.0)) ||
        (last_pushed_stamp_ && sample.stamp_sec <= *last_pushed_stamp_) ||
        (sample.distance_stamp_sec && last_distance_sample_ &&
         *sample.distance_stamp_sec <= *last_distance_sample_->distance_stamp_sec)) {
      return false;
    }
    if (sample.distance_stamp_sec) {
      if (last_distance_sample_) {
        const double dt_sec =
            *sample.distance_stamp_sec - *last_distance_sample_->distance_stamp_sec;
        const double previous_lateral =
            last_distance_sample_->lateral_speed_mps.value_or(0.0);
        const double current_lateral = sample.lateral_speed_mps.value_or(0.0);
        if (dt_sec <= maximum_distance_gap_sec_) {
          cumulative_forward_distance_m_ += 0.5 *
              (last_distance_sample_->forward_speed_mps +
               sample.forward_speed_mps) * dt_sec;
          cumulative_lateral_distance_m_ += 0.5 *
              (previous_lateral + current_lateral) * dt_sec;
        }
      }
      if (!std::isfinite(cumulative_forward_distance_m_) ||
          !std::isfinite(cumulative_lateral_distance_m_)) {
        return false;
      }
      sample.cumulative_forward_distance_m = cumulative_forward_distance_m_;
      sample.cumulative_lateral_distance_m = cumulative_lateral_distance_m_;
      last_distance_sample_ = sample;
    }
    if (samples_.size() == capacity_) samples_.pop_front();
    samples_.push_back(sample);
    last_pushed_stamp_ = sample.stamp_sec;
    return true;
  }

  void reset() {
    samples_.clear();
    latest_selected_.reset();
    last_pushed_stamp_.reset();
    last_distance_sample_.reset();
    cumulative_forward_distance_m_ = 0.0;
    cumulative_lateral_distance_m_ = 0.0;
  }

  std::optional<TimedWheelVelocity> take_for_scan(
      double scan_end_sec, double maximum_age_sec, double maximum_future_sec) {
    if (!std::isfinite(scan_end_sec) || !std::isfinite(maximum_age_sec) ||
        !std::isfinite(maximum_future_sec) || maximum_age_sec <= 0.0 ||
        maximum_future_sec < 0.0) {
      return std::nullopt;
    }

    const double latest_eligible = scan_end_sec + maximum_future_sec;
    while (!samples_.empty() && samples_.front().stamp_sec <= latest_eligible) {
      latest_selected_ = samples_.front();
      samples_.pop_front();
    }
    if (!latest_selected_ || latest_selected_->stamp_sec > latest_eligible ||
        scan_end_sec - latest_selected_->stamp_sec > maximum_age_sec) {
      return std::nullopt;
    }
    return latest_selected_;
  }

 private:
  std::size_t capacity_;
  double maximum_distance_gap_sec_;
  std::deque<TimedWheelVelocity> samples_;
  std::optional<TimedWheelVelocity> latest_selected_;
  std::optional<double> last_pushed_stamp_;
  std::optional<TimedWheelVelocity> last_distance_sample_;
  double cumulative_forward_distance_m_{0.0};
  double cumulative_lateral_distance_m_{0.0};
};

}  // namespace fast_lio
