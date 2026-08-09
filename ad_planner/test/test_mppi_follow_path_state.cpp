#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nav_msgs/msg/path.hpp>

#include "ad_planner/local_planning/mppi_nav2/mppi_follow_path_goal_handle_registry.hpp"
#include "ad_planner/local_planning/mppi_nav2/mppi_follow_path_state.hpp"

namespace {

using ad_planner::canonicalize_mppi_follow_path;
using ad_planner::MppiFollowPathAction;
using ad_planner::MppiFollowPathActionType;
using ad_planner::MppiFollowPathCanonicalPath;
using ad_planner::MppiFollowPathEffects;
using ad_planner::MppiFollowPathGoalHandleRegistry;
using ad_planner::MppiFollowPathResultCode;
using ad_planner::MppiFollowPathState;
using ad_planner::MppiFollowPathStateConfig;

nav_msgs::msg::Path path_message(const double first_x = 1.0) {
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.header.stamp.sec = 1;
  for (std::size_t index = 0U; index < 2U; ++index) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = index == 0U ? "" : "odom";
    pose.header.stamp.sec = 1;
    pose.pose.position.x = first_x + static_cast<double>(index);
    pose.pose.position.y = 2.0 + static_cast<double>(index);
    pose.pose.position.z = 0.25;
    pose.pose.orientation.z = std::sin(0.1);
    pose.pose.orientation.w = std::cos(0.1);
    path.poses.push_back(pose);
  }
  return path;
}

MppiFollowPathCanonicalPath canonical_path(const double first_x = 1.0) {
  const auto admission =
      canonicalize_mppi_follow_path(path_message(first_x), 10U);
  EXPECT_TRUE(admission.valid) << admission.reason;
  return admission.path;
}

void expect_actions(
    const MppiFollowPathEffects &effects,
    const std::vector<std::pair<MppiFollowPathActionType, std::uint64_t>>
        &expected) {
  ASSERT_EQ(effects.actions.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    EXPECT_EQ(effects.actions[index].type, expected[index].first);
    EXPECT_EQ(effects.actions[index].generation, expected[index].second);
  }
}

std::uint64_t observe_new(MppiFollowPathState &state,
                          const MppiFollowPathCanonicalPath &path,
                          const std::int64_t receipt_steady_ns) {
  const auto effects = state.observe_path(path, receipt_steady_ns);
  EXPECT_TRUE(effects.new_path_generation.has_value());
  expect_actions(effects, {});
  return effects.new_path_generation.value_or(0U);
}

std::uint64_t send_waiting_goal(MppiFollowPathState &state,
                                const std::int64_t now_steady_ns) {
  const auto effects = state.poll(now_steady_ns, true);
  EXPECT_FALSE(effects.new_path_generation.has_value());
  if (effects.actions.size() != 1U) {
    ADD_FAILURE() << "expected exactly one send action";
    return 0U;
  }
  EXPECT_EQ(effects.actions.front().type, MppiFollowPathActionType::kSendGoal);
  return effects.actions.front().generation;
}

TEST(MppiFollowPathState, CanonicalPathIgnoresStampsButComparesAllContent) {
  auto original = path_message();
  auto republished = original;
  republished.header.stamp.sec = 999;
  republished.header.stamp.nanosec = 42U;
  for (auto &pose : republished.poses) {
    pose.header.stamp.sec = 777;
    pose.header.stamp.nanosec = 5U;
  }

  const auto first = canonicalize_mppi_follow_path(original, 10U);
  const auto second = canonicalize_mppi_follow_path(republished, 10U);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(second.valid);
  EXPECT_EQ(first.path, second.path);

  republished.poses.front().pose.position.x += 0.001;
  const auto moved = canonicalize_mppi_follow_path(republished, 10U);
  ASSERT_TRUE(moved.valid);
  EXPECT_NE(first.path, moved.path);

  republished = original;
  republished.poses.back().pose.orientation.z =
      -republished.poses.back().pose.orientation.z;
  const auto rotated = canonicalize_mppi_follow_path(republished, 10U);
  ASSERT_TRUE(rotated.valid);
  EXPECT_NE(first.path, rotated.path);
}

TEST(MppiFollowPathState, RejectsMalformedAndOversizedPaths) {
  const auto expect_invalid = [](const nav_msgs::msg::Path &path,
                                 const std::size_t max) {
    const auto admission = canonicalize_mppi_follow_path(path, max);
    EXPECT_FALSE(admission.valid);
    EXPECT_FALSE(admission.reason.empty());
    EXPECT_TRUE(admission.path.points.empty());
  };

  auto malformed = path_message();
  malformed.header.frame_id.clear();
  expect_invalid(malformed, 10U);

  malformed = path_message();
  malformed.poses.clear();
  expect_invalid(malformed, 10U);

  malformed = path_message();
  expect_invalid(malformed, 1U);
  expect_invalid(malformed, 0U);

  malformed = path_message();
  malformed.poses.front().header.frame_id = "map";
  expect_invalid(malformed, 10U);

  malformed = path_message();
  malformed.poses.front().pose.position.y =
      std::numeric_limits<double>::quiet_NaN();
  expect_invalid(malformed, 10U);

  malformed = path_message();
  malformed.poses.front().pose.orientation.w =
      std::numeric_limits<double>::infinity();
  expect_invalid(malformed, 10U);

  malformed = path_message();
  malformed.poses.front().pose.orientation.w = 2.0;
  malformed.poses.front().pose.orientation.z = 0.0;
  expect_invalid(malformed, 10U);
}

TEST(MppiFollowPathState, IdenticalRouteSendsOnceAndChangedRoutePreempts) {
  MppiFollowPathState state(MppiFollowPathStateConfig{200});
  const auto first_path = canonical_path();
  const auto first_generation = observe_new(state, first_path, 100);

  expect_actions(state.poll(100, false), {});
  expect_actions(state.poll(110, false), {});
  EXPECT_EQ(send_waiting_goal(state, 120), first_generation);
  expect_actions(state.on_goal_response(first_generation, true), {});

  const auto republished = state.observe_path(first_path, 150);
  EXPECT_FALSE(republished.new_path_generation.has_value());
  expect_actions(republished, {});
  expect_actions(state.poll(150, true), {});

  const auto changed_generation =
      observe_new(state, canonical_path(1.001), 160);
  const auto preempt = state.poll(160, true);
  expect_actions(preempt,
                 {{MppiFollowPathActionType::kSendGoal, changed_generation}});
  EXPECT_NE(changed_generation, first_generation);
}

TEST(MppiFollowPathState, StaleAndInvalidInputCancelCurrentExactlyOnce) {
  MppiFollowPathState state(MppiFollowPathStateConfig{200});
  const auto generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), generation);
  expect_actions(state.on_goal_response(generation, true), {});

  expect_actions(state.poll(300, true), {});
  expect_actions(state.poll(301, true),
                 {{MppiFollowPathActionType::kCancelGoal, generation}});
  expect_actions(state.poll(302, true), {});
  expect_actions(state.deactivate(), {});

  const auto next_generation = observe_new(state, canonical_path(), 400);
  ASSERT_EQ(send_waiting_goal(state, 400), next_generation);
  expect_actions(state.on_goal_response(next_generation, true), {});
  expect_actions(state.observe_path(canonical_path(), 0),
                 {{MppiFollowPathActionType::kCancelGoal, next_generation}});
  expect_actions(state.deactivate(), {});
}

TEST(MppiFollowPathState,
     LateAcceptedObsoleteGenerationIsCanceledOnceAndCannotReplaceCurrent) {
  MppiFollowPathState state(MppiFollowPathStateConfig{500});
  const auto old_generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), old_generation);

  const auto current_generation = observe_new(state, canonical_path(2.0), 110);
  expect_actions(state.poll(110, true), {});

  const auto old_response = state.on_goal_response(old_generation, true);
  EXPECT_TRUE(old_response.goal_response_was_pending);
  expect_actions(old_response,
                 {{MppiFollowPathActionType::kCancelGoal, old_generation}});
  const auto duplicate_old_response =
      state.on_goal_response(old_generation, true);
  EXPECT_FALSE(duplicate_old_response.goal_response_was_pending);
  expect_actions(duplicate_old_response, {});

  ASSERT_EQ(send_waiting_goal(state, 120), current_generation);
  const auto current_response =
      state.on_goal_response(current_generation, true);
  EXPECT_TRUE(current_response.goal_response_was_pending);
  expect_actions(current_response, {});

  const auto before_old_result = state.snapshot();
  ASSERT_TRUE(before_old_result.current_generation.has_value());
  ASSERT_TRUE(before_old_result.accepted_generation.has_value());
  EXPECT_EQ(*before_old_result.current_generation, current_generation);
  EXPECT_EQ(*before_old_result.accepted_generation, current_generation);

  expect_actions(
      state.on_goal_result(old_generation, MppiFollowPathResultCode::kAborted),
      {});
  const auto after_old_result = state.snapshot();
  EXPECT_EQ(after_old_result.current_generation, current_generation);
  EXPECT_EQ(after_old_result.accepted_generation, current_generation);
}

TEST(MppiFollowPathGoalHandleRegistry,
     RapidAcceptedReplacementsRemainBoundedWithoutResultCallbacks) {
  MppiFollowPathGoalHandleRegistry<std::shared_ptr<int>> registry;
  for (std::uint64_t generation = 1U; generation <= 100U; ++generation) {
    registry.record_accepted_response(
        generation, std::make_shared<int>(static_cast<int>(generation)), true,
        false, generation);
    EXPECT_EQ(registry.size(), 1U);
    EXPECT_EQ(registry.current_generation(), generation);
  }
}

TEST(MppiFollowPathGoalHandleRegistry,
     UnknownAndDuplicateResponsesDoNotLeakOrReplaceHandles) {
  MppiFollowPathGoalHandleRegistry<std::shared_ptr<int>> registry;
  const auto original = std::make_shared<int>(1);
  registry.record_accepted_response(1U, original, true, false, 1U);

  registry.record_accepted_response(999U, std::make_shared<int>(999), false,
                                    true, 1U);
  registry.record_accepted_response(1U, std::make_shared<int>(2), false, false,
                                    1U);
  EXPECT_EQ(registry.size(), 1U);
  EXPECT_FALSE(registry.contains(999U));

  const auto retained = registry.take_for_cancel(1U);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(*retained, original);
  EXPECT_EQ(registry.size(), 0U);
}

TEST(MppiFollowPathGoalHandleRegistry,
     StaleAcceptedHandleExistsOnlyUntilCancelWorkIsCollected) {
  MppiFollowPathGoalHandleRegistry<std::shared_ptr<int>> registry;
  registry.record_accepted_response(1U, std::make_shared<int>(1), true, false,
                                    1U);
  registry.record_accepted_response(2U, std::make_shared<int>(2), true, true,
                                    1U);

  EXPECT_EQ(registry.size(), 2U);
  EXPECT_EQ(registry.current_generation(), 1U);
  const auto stale = registry.take_for_cancel(2U);
  ASSERT_TRUE(stale.has_value());
  EXPECT_EQ(**stale, 2);
  EXPECT_EQ(registry.size(), 1U);
  EXPECT_EQ(registry.current_generation(), 1U);
}

TEST(MppiFollowPathState,
     LateAcceptedReplacementAfterDeactivationIsCanceledOnceAndCannotRevive) {
  MppiFollowPathState state(MppiFollowPathStateConfig{500});
  const auto active_generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), active_generation);
  expect_actions(state.on_goal_response(active_generation, true), {});

  const auto pending_replacement = observe_new(state, canonical_path(2.0), 110);
  ASSERT_EQ(send_waiting_goal(state, 110), pending_replacement);
  ASSERT_EQ(state.snapshot().pending_goal_response_count, 1U);

  expect_actions(state.deactivate(),
                 {{MppiFollowPathActionType::kCancelGoal, active_generation}});
  expect_actions(state.deactivate(), {});
  expect_actions(state.poll(120, true), {});

  expect_actions(
      state.on_goal_response(pending_replacement, true),
      {{MppiFollowPathActionType::kCancelGoal, pending_replacement}});
  expect_actions(state.on_goal_response(pending_replacement, true), {});
  expect_actions(state.poll(130, true), {});
  expect_actions(state.on_goal_result(pending_replacement,
                                      MppiFollowPathResultCode::kCanceled),
                 {});

  const auto terminal = state.snapshot();
  EXPECT_FALSE(terminal.has_path);
  EXPECT_FALSE(terminal.current_generation.has_value());
  EXPECT_FALSE(terminal.accepted_generation.has_value());
  EXPECT_EQ(terminal.pending_goal_response_count, 0U);
}

TEST(MppiFollowPathState,
     RapidPathChangesBoundPendingResponseAndSendOnlyLatestGeneration) {
  MppiFollowPathState state(MppiFollowPathStateConfig{500});
  const auto first_generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), first_generation);
  ASSERT_EQ(state.snapshot().pending_goal_response_count, 1U);

  const auto skipped_generation = observe_new(state, canonical_path(2.0), 110);
  expect_actions(state.poll(110, true), {});
  EXPECT_EQ(state.snapshot().pending_goal_response_count, 1U);

  const auto latest_generation = observe_new(state, canonical_path(3.0), 120);
  expect_actions(state.poll(120, true), {});
  EXPECT_EQ(state.snapshot().pending_goal_response_count, 1U);

  expect_actions(state.on_goal_response(first_generation, true),
                 {{MppiFollowPathActionType::kCancelGoal, first_generation}});
  EXPECT_EQ(state.snapshot().pending_goal_response_count, 0U);

  expect_actions(state.poll(130, true),
                 {{MppiFollowPathActionType::kSendGoal, latest_generation}});
  EXPECT_EQ(state.snapshot().pending_goal_response_count, 1U);
  EXPECT_NE(latest_generation, skipped_generation);
}

TEST(MppiFollowPathState, RejectedReplacementCancelsOlderAcceptedGoal) {
  MppiFollowPathState state(MppiFollowPathStateConfig{500});
  const auto old_generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), old_generation);
  expect_actions(state.on_goal_response(old_generation, true), {});

  const auto replacement = observe_new(state, canonical_path(3.0), 110);
  ASSERT_EQ(send_waiting_goal(state, 110), replacement);
  expect_actions(state.on_goal_response(replacement, false),
                 {{MppiFollowPathActionType::kCancelGoal, old_generation}});
  expect_actions(state.on_goal_response(replacement, false), {});

  const auto snapshot = state.snapshot();
  EXPECT_FALSE(snapshot.has_path);
  EXPECT_FALSE(snapshot.current_generation.has_value());
  EXPECT_FALSE(snapshot.accepted_generation.has_value());
}

TEST(MppiFollowPathState,
     NativePreemptionAbortAndExplicitCancellationRemainDistinct) {
  MppiFollowPathState state(MppiFollowPathStateConfig{500});
  const auto old_generation = observe_new(state, canonical_path(), 100);
  ASSERT_EQ(send_waiting_goal(state, 100), old_generation);
  expect_actions(state.on_goal_response(old_generation, true), {});

  const auto replacement = observe_new(state, canonical_path(4.0), 110);
  ASSERT_EQ(send_waiting_goal(state, 110), replacement);
  expect_actions(state.on_goal_response(replacement, true), {});
  expect_actions(
      state.on_goal_result(old_generation, MppiFollowPathResultCode::kAborted),
      {});
  EXPECT_EQ(state.snapshot().accepted_generation, replacement);

  expect_actions(state.deactivate(),
                 {{MppiFollowPathActionType::kCancelGoal, replacement}});
  expect_actions(
      state.on_goal_result(replacement, MppiFollowPathResultCode::kCanceled),
      {});
  expect_actions(state.deactivate(), {});
}

TEST(MppiFollowPathState, MatchingTerminalResultClearsOnlyItsGeneration) {
  for (const auto result_code : {MppiFollowPathResultCode::kSucceeded,
                                 MppiFollowPathResultCode::kAborted,
                                 MppiFollowPathResultCode::kCanceled}) {
    MppiFollowPathState state(MppiFollowPathStateConfig{500});
    const auto generation = observe_new(state, canonical_path(), 100);
    ASSERT_EQ(send_waiting_goal(state, 100), generation);
    expect_actions(state.on_goal_response(generation, true), {});

    expect_actions(state.on_goal_result(generation, result_code), {});
    const auto snapshot = state.snapshot();
    EXPECT_FALSE(snapshot.has_path);
    EXPECT_FALSE(snapshot.current_generation.has_value());
    EXPECT_FALSE(snapshot.accepted_generation.has_value());
  }
}

TEST(MppiFollowPathState, RejectsInvalidTimeoutConfiguration) {
  EXPECT_THROW((void)MppiFollowPathState{MppiFollowPathStateConfig{0}},
               std::invalid_argument);
  EXPECT_THROW((void)MppiFollowPathState{MppiFollowPathStateConfig{-1}},
               std::invalid_argument);
}

} // namespace
