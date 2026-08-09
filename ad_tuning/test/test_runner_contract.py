from collections.abc import Mapping

from ad_tuning.objective import RunMetrics
from ad_tuning.runner import TrialRunner


class FakeRunner:
    def run_trial(self, parameters: Mapping[str, float]) -> RunMetrics:
        return RunMetrics(
            completed=True,
            elapsed_s=10.0,
            progress_m=100.0,
            mean_cte_sq_m2=0.04,
            distance_cte_mse_m2=0.04,
            time_cte_mse_m2=0.04,
            max_cte_m=0.4,
            rear_mean_cte_sq_m2=0.04,
            rear_distance_cte_mse_m2=0.04,
            rear_time_cte_mse_m2=0.04,
            rear_max_cte_m=0.4,
            overspeed_s=0.0,
        )


def test_fake_runner_satisfies_the_runtime_contract():
    runner = FakeRunner()
    assert isinstance(runner, TrialRunner)
    assert runner.run_trial({"stanley.lookahead_m": 5.0}).completed
