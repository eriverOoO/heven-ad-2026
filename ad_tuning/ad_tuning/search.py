from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Mapping

import optuna


PROFILE_STANLEY_STUDY_VERSION = "profile_stanley_ros_morai_v4"
PROFILE_STANLEY_PARAMETER_NAMES = (
    "profile_stanley.cross_track_gain",
    "profile_stanley.speed_softening_mps",
    "profile_stanley.heading_error_gain",
    "profile_stanley.lookahead_time_s",
    "profile_stanley.curvature_lookahead_m",
    "profile_stanley.speed_pid.kp",
    "profile_stanley.speed_pid.kd",
    "profile_stanley.brake_pid.kp",
    "profile_stanley.brake_pid.kd",
)
PROFILE_STANLEY_WARM_STARTS = (
    {
        "profile_stanley.cross_track_gain": 0.91,
        "profile_stanley.speed_softening_mps": 2.57,
        "profile_stanley.heading_error_gain": 1.0,
        "profile_stanley.lookahead_time_s": 0.16,
        "profile_stanley.curvature_lookahead_m": 1.0,
        "profile_stanley.speed_pid.kp": 1.08,
        "profile_stanley.speed_pid.kd": 0.036,
        "profile_stanley.brake_pid.kp": 0.20,
        "profile_stanley.brake_pid.kd": 0.010,
    },
    {
        "profile_stanley.cross_track_gain": 0.75,
        "profile_stanley.speed_softening_mps": 2.8,
        "profile_stanley.heading_error_gain": 0.9,
        "profile_stanley.lookahead_time_s": 0.20,
        "profile_stanley.curvature_lookahead_m": 1.5,
        "profile_stanley.speed_pid.kp": 0.85,
        "profile_stanley.speed_pid.kd": 0.025,
        "profile_stanley.brake_pid.kp": 0.10,
        "profile_stanley.brake_pid.kd": 0.005,
    },
    {
        "profile_stanley.cross_track_gain": 1.1,
        "profile_stanley.speed_softening_mps": 2.2,
        "profile_stanley.heading_error_gain": 1.1,
        "profile_stanley.lookahead_time_s": 0.12,
        "profile_stanley.curvature_lookahead_m": 2.0,
        "profile_stanley.speed_pid.kp": 1.20,
        "profile_stanley.speed_pid.kd": 0.050,
        "profile_stanley.brake_pid.kp": 0.35,
        "profile_stanley.brake_pid.kd": 0.015,
    },
)

DWA_STUDY_VERSION = "dwa_ros_morai_v4"
DWA_PARAMETER_NAMES = (
    "dwa.goal_weight",
    "dwa.heading_weight",
    "dwa.clearance_weight",
    "dwa.smoothness_weight",
    "dwa.path_distance_weight",
    "dwa.speed_weight",
    "dwa.speed_pid.kp",
    "dwa.speed_pid.kd",
    "dwa.brake_pid.kp",
    "dwa.brake_pid.kd",
)
DWA_WARM_STARTS = (
    {
        "dwa.goal_weight": 0.5,
        "dwa.heading_weight": 1.0,
        "dwa.clearance_weight": 2.0,
        "dwa.smoothness_weight": 0.15,
        "dwa.path_distance_weight": 1.5,
        "dwa.speed_weight": 0.5,
        "dwa.speed_pid.kp": 0.30,
        "dwa.speed_pid.kd": 0.01,
        "dwa.brake_pid.kp": 0.25,
        "dwa.brake_pid.kd": 0.01,
    },
    {
        "dwa.goal_weight": 0.3,
        "dwa.heading_weight": 1.2,
        "dwa.clearance_weight": 4.0,
        "dwa.smoothness_weight": 0.08,
        "dwa.path_distance_weight": 2.5,
        "dwa.speed_weight": 0.25,
        "dwa.speed_pid.kp": 0.25,
        "dwa.speed_pid.kd": 0.02,
        "dwa.brake_pid.kp": 0.30,
        "dwa.brake_pid.kd": 0.02,
    },
    {
        "dwa.goal_weight": 0.8,
        "dwa.heading_weight": 2.0,
        "dwa.clearance_weight": 1.5,
        "dwa.smoothness_weight": 0.03,
        "dwa.path_distance_weight": 3.5,
        "dwa.speed_weight": 1.0,
        "dwa.speed_pid.kp": 0.45,
        "dwa.speed_pid.kd": 0.015,
        "dwa.brake_pid.kp": 0.50,
        "dwa.brake_pid.kd": 0.005,
    },
)

INHERITANCE_CONTEXT_KEYS = (
    "scenario",
    "weather",
    "morai_version",
    "vehicle_profile_id",
    "timing_source",
)


def normalize_storage_url(
    output_dir: Path,
    storage_url: str = "",
    *,
    sqlite_filename: str = "profile_stanley_optuna.sqlite3",
) -> str:
    """Resolve a PostgreSQL URL or the single-worker SQLite fallback."""
    resolved = storage_url.strip()
    if not resolved:
        resolved = os.environ.get("OPTUNA_STORAGE_URL", "").strip()
    if not resolved:
        database = (output_dir / sqlite_filename).resolve()
        return f"sqlite:///{database}"
    if resolved.startswith("postgres://"):
        resolved = "postgresql://" + resolved.removeprefix("postgres://")
    if resolved.startswith("postgresql://"):
        resolved = (
            "postgresql+psycopg://"
            + resolved.removeprefix("postgresql://")
        )
    if not (
        resolved.startswith("postgresql+psycopg://")
        or resolved.startswith("sqlite:///")
    ):
        raise ValueError(
            "storage URL must use postgresql+psycopg:// or sqlite:///"
        )
    return resolved


def redact_storage_url(storage_url: str) -> str:
    """Return a log-safe URL without exposing a database password."""
    return re.sub(r"(://[^:/?#]+:)[^@/?#]*@", r"\1***@", storage_url)


def resolve_worker_id(explicit: str = "") -> str:
    raw = (
        explicit.strip()
        or os.environ.get("AD_TUNING_WORKER_ID", "").strip()
        or socket.gethostname()
    )
    worker_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    if not worker_id:
        raise ValueError("worker_id must contain a safe identifier")
    return worker_id[:80]


def derive_worker_seed(base_seed: int, worker_id: str) -> int:
    worker_hash = int.from_bytes(
        hashlib.sha256(worker_id.encode("utf-8")).digest()[:4], "big"
    )
    return (int(base_seed) ^ worker_hash) & 0x7FFFFFFF


def build_experiment_fingerprint(
    metadata: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Canonicalize all conditions that make trial scores comparable."""
    canonical = json.loads(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20], canonical


def runtime_source_digest(package_dir: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(package_dir.glob("*.py"))
    if not sources:
        raise ValueError(f"no Python sources found under {package_dir}")
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def runtime_artifact_manifest(
    artifacts: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Hash exact installed binaries and configuration used by a worker."""
    manifest: dict[str, dict[str, object]] = {}
    for name, path in sorted(artifacts.items()):
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"required tuning artifact is missing: {name}={resolved}"
            )
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[name] = {
            "sha256": digest.hexdigest(),
            "size_bytes": resolved.stat().st_size,
        }
    return manifest


def recorded_constraints(
    trial: optuna.trial.FrozenTrial,
) -> tuple[float, ...]:
    """Return feasibility violations recorded by the real MORAI objective."""
    values = trial.user_attrs.get("constraint_values")
    if not isinstance(values, (list, tuple)) or not values:
        return (1.0, 1.0)
    return tuple(float(value) for value in values)


def profile_stanley_constraints(
    trial: optuna.trial.FrozenTrial,
) -> tuple[float, ...]:
    return recorded_constraints(trial)


def dwa_constraints(
    trial: optuna.trial.FrozenTrial,
) -> tuple[float, ...]:
    return recorded_constraints(trial)


def _validate_warm_starts(
    parameter_names: tuple[str, ...],
    warm_starts: tuple[dict[str, float], ...],
) -> None:
    expected = set(parameter_names)
    for index, parameters in enumerate(warm_starts):
        actual = set(parameters)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"warm start {index} does not match search space; "
                f"missing={missing}, extra={extra}"
            )


def _inheritance_signature(
    metadata: Mapping[str, object] | None,
) -> str:
    """Keep only conditions that make old parameters relevant as new seeds."""
    if not metadata:
        return ""
    context = metadata.get("context")
    stable_context = (
        {
            name: context.get(name)
            for name in INHERITANCE_CONTEXT_KEYS
        }
        if isinstance(context, Mapping)
        else {}
    )
    relevant = {
        "search_space_version": metadata.get("search_space_version"),
        "route_digest": metadata.get("route_digest"),
        "stack_conditions": metadata.get("stack_conditions"),
        "context": stable_context,
    }
    return json.dumps(
        relevant, sort_keys=True, separators=(",", ":"), default=str
    )


def _trial_seed(
    trial: optuna.trial.FrozenTrial,
    parameter_names: tuple[str, ...],
) -> dict[str, float] | None:
    if (
        trial.state is not optuna.trial.TrialState.COMPLETE
        or trial.user_attrs.get("feasible") is not True
    ):
        return None
    if set(trial.params) != set(parameter_names):
        return None
    parameters = {name: float(trial.params[name]) for name in parameter_names}
    if not all(
        value == value and abs(value) != float("inf")
        for value in parameters.values()
    ):
        return None
    return parameters


def _inherited_warm_starts(
    *,
    storage_url: str,
    current_study_name: str,
    current_metadata: Mapping[str, object] | None,
    study_version: str,
    parameter_names: tuple[str, ...],
    minimum_complete_trials: int,
    top_k: int,
) -> tuple[str, int, tuple[tuple[int, dict[str, float]], ...]]:
    """Select top feasible parameters from the newest mature compatible study."""
    if minimum_complete_trials <= 0 or top_k <= 0:
        return "", 0, ()
    signature = _inheritance_signature(current_metadata)
    if not signature:
        return "", 0, ()

    compatible: list[tuple[str, str, optuna.Study, int]] = []
    summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    for summary in summaries:
        if (
            summary.study_name == current_study_name
            or not summary.study_name.startswith(f"{study_version}_")
        ):
            continue
        metadata = summary.user_attrs.get("experiment_metadata")
        if (
            not isinstance(metadata, Mapping)
            or _inheritance_signature(metadata) != signature
        ):
            continue
        previous = optuna.load_study(
            study_name=summary.study_name, storage=storage_url
        )
        complete_count = sum(
            trial.state is optuna.trial.TrialState.COMPLETE
            for trial in previous.trials
        )
        if complete_count >= minimum_complete_trials:
            compatible.append(
                (
                    str(summary.datetime_start or ""),
                    summary.study_name,
                    previous,
                    complete_count,
                )
            )
    if not compatible:
        return "", 0, ()

    _, source_name, source, complete_count = max(
        compatible, key=lambda item: (item[0], item[1])
    )
    ranked: list[tuple[float, int, dict[str, float]]] = []
    for trial in source.trials:
        parameters = _trial_seed(trial, parameter_names)
        if parameters is None:
            continue
        cost = trial.user_attrs.get("cost", trial.value)
        if not isinstance(cost, (int, float)):
            continue
        value = float(cost)
        if value != value or abs(value) == float("inf"):
            continue
        ranked.append((value, trial.number, parameters))
    ranked.sort(key=lambda item: (item[0], item[1]))

    selected: list[tuple[int, dict[str, float]]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for _, trial_number, parameters in ranked:
        identity = tuple(parameters.items())
        if identity in seen:
            continue
        seen.add(identity)
        selected.append((trial_number, parameters))
        if len(selected) >= top_k:
            break
    return source_name, complete_count, tuple(selected)


@contextmanager
def _warm_start_lock(
    storage_url: str, study_name: str
) -> Iterator[None]:
    if not storage_url.startswith("postgresql+psycopg://"):
        yield
        return

    import psycopg

    connection_url = storage_url.replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    with psycopg.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                (f"ad_tuning_warm_start:{study_name}",),
            )
            try:
                yield
            finally:
                cursor.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (f"ad_tuning_warm_start:{study_name}",),
                )


def initialize_profile_stanley_study(
    study: optuna.Study,
    *,
    storage_url: str,
    worker_id: str,
    single_worker_storage: bool,
    experiment_metadata: Mapping[str, object] | None = None,
    inherit_minimum_complete_trials: int = 36,
    inherit_top_k: int = 5,
) -> None:
    """Queue seeds exactly once under the shared storage lock."""
    _initialize_study(
        study,
        parameter_names=PROFILE_STANLEY_PARAMETER_NAMES,
        warm_starts=PROFILE_STANLEY_WARM_STARTS,
        storage_url=storage_url,
        worker_id=worker_id,
        single_worker_storage=single_worker_storage,
        experiment_metadata=experiment_metadata,
        study_version=PROFILE_STANLEY_STUDY_VERSION,
        inherit_minimum_complete_trials=inherit_minimum_complete_trials,
        inherit_top_k=inherit_top_k,
    )


def initialize_dwa_study(
    study: optuna.Study,
    *,
    storage_url: str,
    worker_id: str,
    single_worker_storage: bool,
    experiment_metadata: Mapping[str, object] | None = None,
    inherit_minimum_complete_trials: int = 40,
    inherit_top_k: int = 5,
) -> None:
    _initialize_study(
        study,
        parameter_names=DWA_PARAMETER_NAMES,
        warm_starts=DWA_WARM_STARTS,
        storage_url=storage_url,
        worker_id=worker_id,
        single_worker_storage=single_worker_storage,
        experiment_metadata=experiment_metadata,
        study_version=DWA_STUDY_VERSION,
        inherit_minimum_complete_trials=inherit_minimum_complete_trials,
        inherit_top_k=inherit_top_k,
    )


def _initialize_study(
    study: optuna.Study,
    *,
    parameter_names: tuple[str, ...],
    warm_starts: tuple[dict[str, float], ...],
    storage_url: str,
    worker_id: str,
    single_worker_storage: bool,
    experiment_metadata: Mapping[str, object] | None,
    study_version: str,
    inherit_minimum_complete_trials: int,
    inherit_top_k: int,
) -> None:
    _validate_warm_starts(parameter_names, warm_starts)
    postgres = storage_url.startswith("postgresql+psycopg://")
    if not single_worker_storage and not postgres:
        raise ValueError(
            "multi-worker warm starts require PostgreSQL advisory locking"
        )
    with _warm_start_lock(storage_url, study.study_name):
        if study.user_attrs.get("warm_start_initialization_complete"):
            return
        source_name, source_complete_count, inherited = (
            _inherited_warm_starts(
                storage_url=storage_url,
                current_study_name=study.study_name,
                current_metadata=experiment_metadata,
                study_version=study_version,
                parameter_names=parameter_names,
                minimum_complete_trials=inherit_minimum_complete_trials,
                top_k=inherit_top_k,
            )
        )
        queued_before = len(study.trials)
        for _, parameters in inherited:
            study.enqueue_trial(parameters, skip_if_exists=True)
        for parameters in warm_starts:
            study.enqueue_trial(parameters, skip_if_exists=True)
        study.set_user_attr("warm_start_coordinator", worker_id)
        study.set_user_attr(
            "warm_start_count", len(study.trials) - queued_before
        )
        study.set_user_attr(
            "inherited_warm_start_count", len(inherited)
        )
        study.set_user_attr(
            "inherited_from_study", source_name
        )
        study.set_user_attr(
            "inherited_source_complete_trials", source_complete_count
        )
        study.set_user_attr(
            "inherited_source_trial_numbers",
            [trial_number for trial_number, _ in inherited],
        )
        study.set_user_attr(
            "inherit_minimum_complete_trials",
            inherit_minimum_complete_trials,
        )
        study.set_user_attr("inherit_top_k", inherit_top_k)
        study.set_user_attr("warm_start_initialization_complete", True)


def _rdb_storage(
    *,
    storage_url: str,
    worker_id: str,
    heartbeat_interval_s: int | None,
    heartbeat_grace_period_s: int | None,
    heartbeat_retry_count: int,
    connect_timeout_s: int,
) -> optuna.storages.RDBStorage:
    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    if storage_url.startswith("postgresql+psycopg://"):
        engine_kwargs.update(
            {
                "pool_recycle": 300,
                "connect_args": {
                    "connect_timeout": connect_timeout_s,
                    "application_name": f"ad_tuning_{worker_id}"[:63],
                },
            }
        )
    retry_callback = None
    if heartbeat_interval_s is not None:
        retry_callback = optuna.storages.RetryHeartbeatStaleTrialCallback(
            max_retry=heartbeat_retry_count
        )
    return optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval_s,
        grace_period=heartbeat_grace_period_s,
        heartbeat_stale_trial_callback=retry_callback,
    )


def create_profile_stanley_study(
    output_dir: Path,
    experiment_fingerprint: str,
    seed: int,
    *,
    storage_url: str = "",
    worker_id: str = "local",
    heartbeat_interval_s: int | None = None,
    heartbeat_grace_period_s: int | None = None,
    heartbeat_retry_count: int = 1,
    connect_timeout_s: int = 3,
    experiment_metadata: Mapping[str, object] | None = None,
    inherit_minimum_complete_trials: int = 36,
    inherit_top_k: int = 5,
) -> optuna.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_url = normalize_storage_url(output_dir, storage_url)
    sampler = optuna.samplers.TPESampler(
        seed=derive_worker_seed(seed, worker_id),
        n_startup_trials=36,
        n_ei_candidates=64,
        multivariate=True,
        constant_liar=True,
        constraints_func=profile_stanley_constraints,
    )
    storage = _rdb_storage(
        storage_url=resolved_url,
        worker_id=worker_id,
        heartbeat_interval_s=heartbeat_interval_s,
        heartbeat_grace_period_s=heartbeat_grace_period_s,
        heartbeat_retry_count=heartbeat_retry_count,
        connect_timeout_s=connect_timeout_s,
    )
    study = optuna.create_study(
        study_name=(
            f"{PROFILE_STANLEY_STUDY_VERSION}_{experiment_fingerprint}"
        ),
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    stored_fingerprint = study.user_attrs.get("experiment_fingerprint")
    if (
        stored_fingerprint is not None
        and stored_fingerprint != experiment_fingerprint
    ):
        raise ValueError(
            "stored experiment fingerprint differs from this worker"
        )
    if stored_fingerprint is None:
        study.set_user_attr(
            "experiment_fingerprint", experiment_fingerprint
        )
    if experiment_metadata is not None:
        stored_metadata = study.user_attrs.get("experiment_metadata")
        metadata = dict(experiment_metadata)
        if stored_metadata is not None and stored_metadata != metadata:
            raise ValueError(
                "stored experiment metadata differs from this worker"
            )
        if stored_metadata is None:
            study.set_user_attr("experiment_metadata", metadata)
    study.set_user_attr(
        "search_space_version", PROFILE_STANLEY_STUDY_VERSION
    )
    single_worker_storage = resolved_url.startswith("sqlite:///")
    optuna.storages.fail_stale_trials(study)
    initialize_profile_stanley_study(
        study,
        storage_url=resolved_url,
        worker_id=worker_id,
        single_worker_storage=single_worker_storage,
        experiment_metadata=experiment_metadata,
        inherit_minimum_complete_trials=inherit_minimum_complete_trials,
        inherit_top_k=inherit_top_k,
    )
    return study


def create_dwa_study(
    output_dir: Path,
    experiment_fingerprint: str,
    seed: int,
    *,
    storage_url: str = "",
    worker_id: str = "local",
    heartbeat_interval_s: int | None = None,
    heartbeat_grace_period_s: int | None = None,
    heartbeat_retry_count: int = 1,
    connect_timeout_s: int = 3,
    experiment_metadata: Mapping[str, object] | None = None,
    inherit_minimum_complete_trials: int = 40,
    inherit_top_k: int = 5,
) -> optuna.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_url = normalize_storage_url(
        output_dir,
        storage_url,
        sqlite_filename="dwa_optuna.sqlite3",
    )
    sampler = optuna.samplers.TPESampler(
        seed=derive_worker_seed(seed, worker_id),
        n_startup_trials=40,
        n_ei_candidates=64,
        multivariate=True,
        constant_liar=True,
        constraints_func=dwa_constraints,
    )
    storage = _rdb_storage(
        storage_url=resolved_url,
        worker_id=worker_id,
        heartbeat_interval_s=heartbeat_interval_s,
        heartbeat_grace_period_s=heartbeat_grace_period_s,
        heartbeat_retry_count=heartbeat_retry_count,
        connect_timeout_s=connect_timeout_s,
    )
    study = optuna.create_study(
        study_name=f"{DWA_STUDY_VERSION}_{experiment_fingerprint}",
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    stored_fingerprint = study.user_attrs.get("experiment_fingerprint")
    if (
        stored_fingerprint is not None
        and stored_fingerprint != experiment_fingerprint
    ):
        raise ValueError(
            "stored experiment fingerprint differs from this worker"
        )
    if stored_fingerprint is None:
        study.set_user_attr(
            "experiment_fingerprint", experiment_fingerprint
        )
    if experiment_metadata is not None:
        stored_metadata = study.user_attrs.get("experiment_metadata")
        metadata = dict(experiment_metadata)
        if stored_metadata is not None and stored_metadata != metadata:
            raise ValueError(
                "stored experiment metadata differs from this worker"
            )
        if stored_metadata is None:
            study.set_user_attr("experiment_metadata", metadata)
    study.set_user_attr("search_space_version", DWA_STUDY_VERSION)
    single_worker_storage = resolved_url.startswith("sqlite:///")
    optuna.storages.fail_stale_trials(study)
    initialize_dwa_study(
        study,
        storage_url=resolved_url,
        worker_id=worker_id,
        single_worker_storage=single_worker_storage,
        experiment_metadata=experiment_metadata,
        inherit_minimum_complete_trials=inherit_minimum_complete_trials,
        inherit_top_k=inherit_top_k,
    )
    return study


def suggest_profile_stanley_parameters(
    trial: optuna.Trial,
) -> dict[str, float]:
    return {
        "profile_stanley.cross_track_gain": trial.suggest_float(
            "profile_stanley.cross_track_gain", 0.5, 1.5
        ),
        "profile_stanley.speed_softening_mps": trial.suggest_float(
            "profile_stanley.speed_softening_mps", 1.0, 4.0
        ),
        "profile_stanley.heading_error_gain": trial.suggest_float(
            "profile_stanley.heading_error_gain", 0.7, 1.3
        ),
        "profile_stanley.lookahead_time_s": trial.suggest_float(
            "profile_stanley.lookahead_time_s", 0.08, 0.30
        ),
        "profile_stanley.curvature_lookahead_m": trial.suggest_float(
            "profile_stanley.curvature_lookahead_m", 0.75, 3.0
        ),
        "profile_stanley.speed_pid.kp": trial.suggest_float(
            "profile_stanley.speed_pid.kp", 0.05, 1.2, log=True
        ),
        "profile_stanley.speed_pid.kd": trial.suggest_float(
            "profile_stanley.speed_pid.kd", 0.0, 0.10
        ),
        "profile_stanley.brake_pid.kp": trial.suggest_float(
            "profile_stanley.brake_pid.kp", 0.02, 0.60, log=True
        ),
        "profile_stanley.brake_pid.kd": trial.suggest_float(
            "profile_stanley.brake_pid.kd", 0.0, 0.05
        ),
    }


def suggest_dwa_parameters(trial: optuna.Trial) -> dict[str, float]:
    return {
        "dwa.goal_weight": trial.suggest_float(
            "dwa.goal_weight", 0.1, 2.0, log=True
        ),
        "dwa.heading_weight": trial.suggest_float(
            "dwa.heading_weight", 0.1, 4.0, log=True
        ),
        "dwa.clearance_weight": trial.suggest_float(
            "dwa.clearance_weight", 0.5, 8.0, log=True
        ),
        "dwa.smoothness_weight": trial.suggest_float(
            "dwa.smoothness_weight", 0.01, 0.5, log=True
        ),
        "dwa.path_distance_weight": trial.suggest_float(
            "dwa.path_distance_weight", 0.25, 8.0, log=True
        ),
        "dwa.speed_weight": trial.suggest_float(
            "dwa.speed_weight", 0.05, 3.0, log=True
        ),
        "dwa.speed_pid.kp": trial.suggest_float(
            "dwa.speed_pid.kp", 0.05, 1.2, log=True
        ),
        "dwa.speed_pid.kd": trial.suggest_float(
            "dwa.speed_pid.kd", 0.0, 0.1
        ),
        # With the 0.5 s dynamic window, 1.8 m/s^2 model, and 0.1 m/s
        # deadband, Kp=0.25 commands about 20% brake for the largest
        # one-cycle speed reduction. Optuna may only strengthen that response.
        "dwa.brake_pid.kp": trial.suggest_float(
            "dwa.brake_pid.kp", 0.25, 0.80, log=True
        ),
        "dwa.brake_pid.kd": trial.suggest_float(
            "dwa.brake_pid.kd", 0.0, 0.05
        ),
    }
