from heven_slam.slam_health import SlamHealthGuard


def test_pose_jump_is_rejected() -> None:
    guard = SlamHealthGuard(max_pose_step_m=1.0, max_odom_speed_mps=100.0)

    assert guard.observe_odom(0.0, 0.0, 1.0) is None
    assert guard.observe_odom(0.2, 0.1, 1.1) is None
    assert "jumped" in guard.observe_odom(2.0, 0.1, 1.2)


def test_impossible_odom_velocity_is_rejected() -> None:
    guard = SlamHealthGuard(max_pose_step_m=1.0, max_odom_speed_mps=2.0)

    assert guard.observe_odom(0.0, 0.0, 1.0) is None
    assert guard.observe_odom(0.1, 0.0, 1.1) is None
    assert "speed" in guard.observe_odom(0.5, 0.0, 1.2)


def test_stale_odom_only_stops_an_armed_moving_vehicle() -> None:
    guard = SlamHealthGuard(
        max_odom_age_sec=1.0,
        moving_speed_threshold_mps=0.15,
        startup_grace_sec=2.0,
    )
    guard.observe_odom(0.0, 0.0, 10.0)
    guard.set_vehicle_speed(0.8)

    assert guard.evaluate(20.0) is None
    guard.set_armed(True, 20.0)
    assert guard.evaluate(21.0) is None
    assert "stale" in guard.evaluate(22.1)


def test_stationary_vehicle_does_not_trigger_stale_guard() -> None:
    guard = SlamHealthGuard(max_odom_age_sec=0.5, startup_grace_sec=0.0)
    guard.set_armed(True, 1.0)
    guard.set_vehicle_speed(0.0)

    assert guard.evaluate(10.0) is None
