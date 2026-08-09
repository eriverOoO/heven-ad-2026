#include <ad_lidar_perception/tracking/imm_predictor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ad_lidar_perception::tracking
{
namespace
{

constexpr std::size_t kModelCount = 3;
constexpr std::size_t kStateSize = 6;
constexpr std::size_t kX = 0;
constexpr std::size_t kY = 1;
constexpr std::size_t kYaw = 2;
constexpr std::size_t kVx = 3;
constexpr std::size_t kVy = 4;
constexpr std::size_t kYawRate = 5;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
constexpr double kMinimumVariance = 1.0e-6;
constexpr double kMinimumProbability = 1.0e-12;

using State = std::array<double, kStateSize>;
using Variance = std::array<double, kStateSize>;

struct ModelState
{
  State mean{};
  Variance variance{};
};

double wrap_angle(double angle) {return std::remainder(angle, kTwoPi);}

double angle_difference(const double lhs, const double rhs)
{
  return wrap_angle(lhs - rhs);
}

std::array<double, kModelCount>
normalized_probabilities(std::array<double, kModelCount> values)
{
  for (double & value : values) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument(
              "IMM probabilities must be finite and non-negative");
    }
  }
  const double total = std::accumulate(values.begin(), values.end(), 0.0);
  if (!(total > 0.0)) {
    throw std::invalid_argument("IMM probabilities must have a positive sum");
  }
  for (double & value : values) {
    value /= total;
  }
  return values;
}

void validate_config(ImmConfig & config)
{
  if (config.horizons_s.empty()) {
    throw std::invalid_argument("IMM prediction horizons must not be empty");
  }
  double previous = 0.0;
  for (const double horizon : config.horizons_s) {
    if (!std::isfinite(horizon) || horizon <= previous) {
      throw std::invalid_argument(
              "IMM horizons must be finite, positive, and increasing");
    }
    previous = horizon;
  }
  config.initial_probabilities =
    normalized_probabilities(config.initial_probabilities);
  for (std::size_t row = 0; row < kModelCount; ++row) {
    std::array<double, kModelCount> row_values{};
    for (std::size_t column = 0; column < kModelCount; ++column) {
      row_values[column] =
        config.transition_probabilities[row * kModelCount + column];
    }
    row_values = normalized_probabilities(row_values);
    for (std::size_t column = 0; column < kModelCount; ++column) {
      config.transition_probabilities[row * kModelCount + column] =
        row_values[column];
    }
  }
  if (!(config.maximum_update_interval_s > 0.0) ||
    !(config.stationary_process_variance > 0.0) ||
    !(config.constant_velocity_process_variance > 0.0) ||
    !(config.coordinated_turn_process_variance > 0.0))
  {
    throw std::invalid_argument(
            "IMM time and process variances must be positive");
  }
}

State measurement_state(const TrackMeasurement2D & measurement)
{
  return {measurement.x_m,
    measurement.y_m,
    wrap_angle(measurement.yaw_rad),
    measurement.vx_world_mps,
    measurement.vy_world_mps,
    measurement.yaw_rate_radps};
}

Variance measurement_variance(const TrackMeasurement2D & measurement)
{
  const auto checked = [](const double value) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
                "IMM measurement variances must be finite and positive");
      }
      return std::max(value, kMinimumVariance);
    };
  return {checked(measurement.position_variance_m2),
    checked(measurement.position_variance_m2),
    checked(measurement.yaw_variance_rad2),
    checked(measurement.velocity_variance_m2ps2),
    checked(measurement.velocity_variance_m2ps2),
    checked(measurement.yaw_rate_variance_rad2ps2)};
}

void propagate(State & state, const MotionModel model, const double dt)
{
  if (model == MotionModel::kStationary) {
    state[kVx] = 0.0;
    state[kVy] = 0.0;
    state[kYawRate] = 0.0;
    return;
  }

  if (model == MotionModel::kConstantVelocity ||
    std::abs(state[kYawRate]) < 1.0e-5)
  {
    state[kX] += state[kVx] * dt;
    state[kY] += state[kVy] * dt;
    if (model == MotionModel::kConstantVelocity) {
      state[kYawRate] = 0.0;
    } else {
      state[kYaw] = wrap_angle(state[kYaw] + state[kYawRate] * dt);
    }
    return;
  }

  const double angle = state[kYawRate] * dt;
  const double sine = std::sin(angle);
  const double cosine = std::cos(angle);
  const double vx = state[kVx];
  const double vy = state[kVy];
  state[kX] += (vx * sine - vy * (1.0 - cosine)) / state[kYawRate];
  state[kY] += (vx * (1.0 - cosine) + vy * sine) / state[kYawRate];
  state[kVx] = cosine * vx - sine * vy;
  state[kVy] = sine * vx + cosine * vy;
  state[kYaw] = wrap_angle(state[kYaw] + angle);
}

double process_variance(const ImmConfig & config, const MotionModel model)
{
  if (model == MotionModel::kStationary) {
    return config.stationary_process_variance;
  }
  if (model == MotionModel::kConstantVelocity) {
    return config.constant_velocity_process_variance;
  }
  return config.coordinated_turn_process_variance;
}

double update_model(
  ModelState & model, const State & observed,
  const Variance & observed_variance,
  const MotionModel motion_model)
{
  double log_likelihood = 0.0;
  for (std::size_t index = 0; index < kStateSize; ++index) {
    double innovation = observed[index] - model.mean[index];
    if (index == kYaw) {
      innovation = angle_difference(observed[index], model.mean[index]);
    }
    const double innovation_variance = std::max(
      model.variance[index] + observed_variance[index], kMinimumVariance);
    log_likelihood += -0.5 * (innovation * innovation / innovation_variance +
      std::log(innovation_variance));
    const double gain = model.variance[index] / innovation_variance;
    model.mean[index] += gain * innovation;
    if (index == kYaw) {
      model.mean[index] = wrap_angle(model.mean[index]);
    }
    model.variance[index] =
      std::max((1.0 - gain) * model.variance[index], kMinimumVariance);
  }

  const double speed_sq =
    observed[kVx] * observed[kVx] + observed[kVy] * observed[kVy];
  if (motion_model == MotionModel::kStationary) {
    log_likelihood -= 0.5 * speed_sq / std::max(observed_variance[kVx], 0.04);
  } else if (motion_model == MotionModel::kConstantVelocity) {
    log_likelihood -= 0.5 * observed[kYawRate] * observed[kYawRate] /
      std::max(observed_variance[kYawRate], 0.01);
  } else {
    // A turn model is useful when a measurable yaw rate exists, but it remains
    // available for nearly straight motion through the Markov transition.
    const double turn_evidence = std::abs(observed[kYawRate]);
    if (turn_evidence < 0.03) {
      log_likelihood -= 1.0;
    }
  }
  return log_likelihood;
}

} // namespace

class ImmPredictor::Impl
{
public:
  explicit Impl(ImmConfig config)
  : config_(std::move(config))
  {
    validate_config(config_);
    probabilities_ = config_.initial_probabilities;
  }

  ImmResult update(
    const TrackMeasurement2D & measurement,
    const std::int64_t stamp_ns)
  {
    if (stamp_ns <= 0) {
      throw std::invalid_argument("IMM measurement time must be positive");
    }
    const State observed = measurement_state(measurement);
    const Variance observed_variance = measurement_variance(measurement);
    for (const double value : observed) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("IMM measurements must be finite");
      }
    }

    if (!initialized_) {
      initialize(observed, observed_variance, stamp_ns);
    } else {
      if (stamp_ns <= last_stamp_ns_) {
        throw std::invalid_argument(
                "IMM measurement time must increase monotonically");
      }
      const double raw_dt =
        static_cast<double>(stamp_ns - last_stamp_ns_) * 1.0e-9;
      const double dt = std::min(raw_dt, config_.maximum_update_interval_s);
      interact_and_predict(dt);
      measurement_update(observed, observed_variance);
      last_stamp_ns_ = stamp_ns;
    }
    return result();
  }

  void reset() noexcept
  {
    initialized_ = false;
    last_stamp_ns_ = 0;
    probabilities_ = config_.initial_probabilities;
    models_ = {};
  }

private:
  void initialize(
    const State & observed, const Variance & observed_variance,
    const std::int64_t stamp_ns)
  {
    for (std::size_t model_index = 0; model_index < kModelCount;
      ++model_index)
    {
      models_[model_index].mean = observed;
      models_[model_index].variance = observed_variance;
      const auto model = static_cast<MotionModel>(model_index);
      if (model == MotionModel::kStationary) {
        models_[model_index].mean[kVx] = 0.0;
        models_[model_index].mean[kVy] = 0.0;
        models_[model_index].mean[kYawRate] = 0.0;
      } else if (model == MotionModel::kConstantVelocity) {
        models_[model_index].mean[kYawRate] = 0.0;
      }
    }

    std::array<double, kModelCount> log_weights{};
    for (std::size_t index = 0; index < kModelCount; ++index) {
      ModelState scoring_model = models_[index];
      log_weights[index] =
        std::log(std::max(probabilities_[index], kMinimumProbability)) +
        update_model(
        scoring_model, observed, observed_variance,
        static_cast<MotionModel>(index));
    }
    normalize_log_weights(log_weights);
    initialized_ = true;
    last_stamp_ns_ = stamp_ns;
  }

  void interact_and_predict(const double dt)
  {
    std::array<ModelState, kModelCount> mixed{};
    std::array<double, kModelCount> predicted_probabilities{};

    for (std::size_t destination = 0; destination < kModelCount;
      ++destination)
    {
      for (std::size_t source = 0; source < kModelCount; ++source) {
        predicted_probabilities[destination] +=
          probabilities_[source] *
          config_
          .transition_probabilities[source * kModelCount + destination];
      }
      const double normalizer =
        std::max(predicted_probabilities[destination], kMinimumProbability);

      double sine_sum = 0.0;
      double cosine_sum = 0.0;
      for (std::size_t source = 0; source < kModelCount; ++source) {
        const double weight =
          probabilities_[source] *
          config_
          .transition_probabilities[source * kModelCount + destination] /
          normalizer;
        for (std::size_t state_index = 0; state_index < kStateSize;
          ++state_index)
        {
          if (state_index != kYaw) {
            mixed[destination].mean[state_index] +=
              weight * models_[source].mean[state_index];
          }
        }
        sine_sum += weight * std::sin(models_[source].mean[kYaw]);
        cosine_sum += weight * std::cos(models_[source].mean[kYaw]);
      }
      mixed[destination].mean[kYaw] = std::atan2(sine_sum, cosine_sum);

      for (std::size_t source = 0; source < kModelCount; ++source) {
        const double weight =
          probabilities_[source] *
          config_
          .transition_probabilities[source * kModelCount + destination] /
          normalizer;
        for (std::size_t state_index = 0; state_index < kStateSize;
          ++state_index)
        {
          const double difference =
            state_index == kYaw ?
            angle_difference(
            models_[source].mean[state_index],
            mixed[destination].mean[state_index]) :
            models_[source].mean[state_index] -
            mixed[destination].mean[state_index];
          mixed[destination].variance[state_index] +=
            weight *
            (models_[source].variance[state_index] + difference * difference);
        }
      }

      const auto model = static_cast<MotionModel>(destination);
      propagate(mixed[destination].mean, model, dt);
      const double noise = process_variance(config_, model) * dt;
      for (double & variance : mixed[destination].variance) {
        variance = std::max(variance + noise, kMinimumVariance);
      }
      if (model == MotionModel::kStationary) {
        mixed[destination].variance[kVx] =
          std::min(mixed[destination].variance[kVx], 0.05);
        mixed[destination].variance[kVy] =
          std::min(mixed[destination].variance[kVy], 0.05);
        mixed[destination].variance[kYawRate] =
          std::min(mixed[destination].variance[kYawRate], 0.02);
      }
    }

    models_ = mixed;
    probabilities_ = normalized_probabilities(predicted_probabilities);
  }

  void measurement_update(
    const State & observed,
    const Variance & observed_variance)
  {
    std::array<double, kModelCount> log_weights{};
    for (std::size_t index = 0; index < kModelCount; ++index) {
      const double likelihood =
        update_model(
        models_[index], observed, observed_variance,
        static_cast<MotionModel>(index));
      log_weights[index] =
        std::log(std::max(probabilities_[index], kMinimumProbability)) +
        likelihood;
    }
    normalize_log_weights(log_weights);
  }

  void
  normalize_log_weights(const std::array<double, kModelCount> & log_weights)
  {
    const double maximum =
      *std::max_element(log_weights.begin(), log_weights.end());
    std::array<double, kModelCount> weights{};
    for (std::size_t index = 0; index < kModelCount; ++index) {
      weights[index] = std::exp(log_weights[index] - maximum);
    }
    probabilities_ = normalized_probabilities(weights);
  }

  ImmKinematicState fused_state() const
  {
    State fused{};
    double sine_sum = 0.0;
    double cosine_sum = 0.0;
    for (std::size_t model = 0; model < kModelCount; ++model) {
      for (std::size_t index = 0; index < kStateSize; ++index) {
        if (index != kYaw) {
          fused[index] += probabilities_[model] * models_[model].mean[index];
        }
      }
      sine_sum += probabilities_[model] * std::sin(models_[model].mean[kYaw]);
      cosine_sum += probabilities_[model] * std::cos(models_[model].mean[kYaw]);
    }
    fused[kYaw] = std::atan2(sine_sum, cosine_sum);
    return {fused[kX], fused[kY], fused[kYaw],
      fused[kVx], fused[kVy], fused[kYawRate]};
  }

  ImmResult result() const
  {
    ImmResult output;
    output.fused_state = fused_state();
    output.model_probabilities = probabilities_;
    output.predicted_states.reserve(config_.horizons_s.size());
    for (const double horizon : config_.horizons_s) {
      std::array<State, kModelCount> predicted{};
      double x = 0.0;
      double y = 0.0;
      double sine_sum = 0.0;
      double cosine_sum = 0.0;
      for (std::size_t model = 0; model < kModelCount; ++model) {
        predicted[model] = models_[model].mean;
        propagate(predicted[model], static_cast<MotionModel>(model), horizon);
        x += probabilities_[model] * predicted[model][kX];
        y += probabilities_[model] * predicted[model][kY];
        sine_sum += probabilities_[model] * std::sin(predicted[model][kYaw]);
        cosine_sum += probabilities_[model] * std::cos(predicted[model][kYaw]);
      }
      output.predicted_states.push_back(
        {horizon, x, y, std::atan2(sine_sum, cosine_sum)});
    }
    return output;
  }

  ImmConfig config_;
  std::array<ModelState, kModelCount> models_{};
  std::array<double, kModelCount> probabilities_{};
  bool initialized_{false};
  std::int64_t last_stamp_ns_{0};
};

ImmPredictor::ImmPredictor(ImmConfig config)
: impl_(std::make_unique<Impl>(std::move(config))) {}

ImmPredictor::~ImmPredictor() = default;
ImmPredictor::ImmPredictor(ImmPredictor &&) noexcept = default;
ImmPredictor & ImmPredictor::operator=(ImmPredictor &&) noexcept = default;
ImmPredictor::ImmPredictor(const ImmPredictor & other)
: impl_(std::make_unique<Impl>(*other.impl_)) {}

ImmPredictor & ImmPredictor::operator=(const ImmPredictor & other)
{
  if (this != &other) {
    impl_ = std::make_unique<Impl>(*other.impl_);
  }
  return *this;
}

ImmResult ImmPredictor::update(
  const TrackMeasurement2D & measurement,
  const std::int64_t stamp_ns)
{
  return impl_->update(measurement, stamp_ns);
}

void ImmPredictor::reset() noexcept {impl_->reset();}

} // namespace ad_lidar_perception::tracking
