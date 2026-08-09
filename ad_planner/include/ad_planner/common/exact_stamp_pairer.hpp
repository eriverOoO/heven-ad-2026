#ifndef AD_PLANNER__COMMON__EXACT_STAMP_PAIRER_HPP_
#define AD_PLANNER__COMMON__EXACT_STAMP_PAIRER_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ad_planner
{

template<typename Left, typename Right>
class ExactStampPairer
{
public:
  struct Pair
  {
    std::int64_t stamp_ns;
    Left left;
    Right right;
  };

  explicit ExactStampPairer(const std::size_t maximum_pending_messages)
  : maximum_pending_messages_(maximum_pending_messages)
  {
    if (maximum_pending_messages_ == 0U) {
      throw std::invalid_argument(
              "exact-stamp pairer capacity must be positive");
    }
  }

  std::optional<Pair> add_left(std::int64_t stamp_ns, Left value)
  {
    validate_stamp(stamp_ns);
    left_.put(stamp_ns, std::move(value), maximum_pending_messages_);
    return match(stamp_ns);
  }

  std::optional<Pair> add_right(std::int64_t stamp_ns, Right value)
  {
    validate_stamp(stamp_ns);
    right_.put(stamp_ns, std::move(value), maximum_pending_messages_);
    return match(stamp_ns);
  }

  std::size_t left_pending() const
  {
    return left_.size();
  }

  std::size_t right_pending() const
  {
    return right_.size();
  }

private:
  template<typename Value>
  class Pending
  {
public:
    void put(
      const std::int64_t stamp_ns, Value value,
      const std::size_t maximum_pending_messages)
    {
      const auto existing = values_.find(stamp_ns);
      if (existing != values_.end()) {
        existing->second = std::move(value);
        return;
      }
      values_.emplace(stamp_ns, std::move(value));
      arrival_order_.push_back(stamp_ns);
      while (values_.size() > maximum_pending_messages) {
        const std::int64_t oldest = arrival_order_.front();
        arrival_order_.pop_front();
        values_.erase(oldest);
      }
    }

    bool contains(const std::int64_t stamp_ns) const
    {
      return values_.find(stamp_ns) != values_.end();
    }

    Value take(const std::int64_t stamp_ns)
    {
      auto found = values_.find(stamp_ns);
      Value value = std::move(found->second);
      values_.erase(found);
      const auto order = std::find(
        arrival_order_.begin(), arrival_order_.end(), stamp_ns);
      arrival_order_.erase(order);
      return value;
    }

    void clear()
    {
      values_.clear();
      arrival_order_.clear();
    }

    std::size_t size() const
    {
      return values_.size();
    }

private:
    std::map<std::int64_t, Value> values_;
    std::deque<std::int64_t> arrival_order_;
  };

  static void validate_stamp(const std::int64_t stamp_ns)
  {
    if (stamp_ns <= 0) {
      throw std::invalid_argument(
              "exact-stamp pairer requires a positive stamp");
    }
  }

  std::optional<Pair> match(const std::int64_t stamp_ns)
  {
    if (!left_.contains(stamp_ns) || !right_.contains(stamp_ns)) {
      return std::nullopt;
    }
    Pair pair{
      stamp_ns,
      left_.take(stamp_ns),
      right_.take(stamp_ns)};
    if (last_paired_stamp_ns_.has_value() &&
      stamp_ns < *last_paired_stamp_ns_)
    {
      left_.clear();
      right_.clear();
    }
    last_paired_stamp_ns_ = stamp_ns;
    return pair;
  }

  std::size_t maximum_pending_messages_;
  Pending<Left> left_;
  Pending<Right> right_;
  std::optional<std::int64_t> last_paired_stamp_ns_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__COMMON__EXACT_STAMP_PAIRER_HPP_
