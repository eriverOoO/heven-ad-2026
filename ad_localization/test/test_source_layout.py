from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_INCLUDE_PREFIX = "ad_localization/" + "global/"
ROLE_DIRECTORIES = (
    "adapter",
    "gnss_imu",
    "imu_quaternion_encoder",
    "quaternion_wheel_gnss_ekf",
    "manager",
)


def _files_below(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _scan_layout(package_root: Path) -> dict[str, list[Path]]:
    legacy_files = [
        path.relative_to(package_root)
        for root in (
            package_root / "include" / "ad_localization" / "global",
            package_root / "src" / "global",
        )
        for path in _files_below(root)
    ]

    missing_role_directories = []
    empty_role_directories = []
    for root in (
        package_root / "include" / "ad_localization",
        package_root / "src",
    ):
        for role in ROLE_DIRECTORIES:
            role_directory = root / role
            relative_role_directory = role_directory.relative_to(package_root)
            if not role_directory.is_dir():
                missing_role_directories.append(relative_role_directory)
            elif not _files_below(role_directory):
                empty_role_directories.append(relative_role_directory)

    stale_references = []
    for root in (
        package_root / "include",
        package_root / "src",
        package_root / "test",
    ):
        for path in _files_below(root):
            if path.suffix not in {".cpp", ".hpp", ".py"}:
                continue
            if LEGACY_INCLUDE_PREFIX in path.read_text(encoding="utf-8"):
                stale_references.append(path.relative_to(package_root))

    return {
        "legacy_files": legacy_files,
        "missing_role_directories": missing_role_directories,
        "empty_role_directories": empty_role_directories,
        "stale_references": stale_references,
    }


def test_global_sources_are_grouped_into_nonempty_role_directories():
    scan = _scan_layout(PACKAGE_ROOT)
    assert not scan["legacy_files"], (
        f"production files remain under global/: {scan['legacy_files']}"
    )
    assert not scan["missing_role_directories"], (
        f"missing role directories: {scan['missing_role_directories']}"
    )
    assert not scan["empty_role_directories"], (
        f"empty role directories: {scan['empty_role_directories']}"
    )


def test_production_and_tests_do_not_include_the_legacy_global_path():
    scan = _scan_layout(PACKAGE_ROOT)
    assert not scan["stale_references"], (
        "legacy global include references remain: "
        f"{scan['stale_references']}"
    )


def test_layout_scan_is_scoped_to_the_localization_package_root(tmp_path):
    localization_root = tmp_path / "ad_localization"
    for role in ROLE_DIRECTORIES:
        header = localization_root / "include" / "ad_localization" / role / "unit.hpp"
        source = localization_root / "src" / role / "unit.cpp"
        header.parent.mkdir(parents=True, exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("#pragma once\n", encoding="utf-8")
        source.write_text("int unit = 0;\n", encoding="utf-8")

    local_violation = localization_root / "src" / "adapter" / "unit.cpp"
    local_violation.write_text(
        f'#include "{LEGACY_INCLUDE_PREFIX}localization_adapter.hpp"\n',
        encoding="utf-8",
    )

    camera_violation = (
        tmp_path / "ad_camera_perception" / "src" / "global" / "camera.cpp"
    )
    camera_violation.parent.mkdir(parents=True)
    camera_contents = f'#include "{LEGACY_INCLUDE_PREFIX}camera.hpp"\n'
    camera_violation.write_text(camera_contents, encoding="utf-8")

    assert _scan_layout(localization_root) == {
        "legacy_files": [],
        "missing_role_directories": [],
        "empty_role_directories": [],
        "stale_references": [Path("src/adapter/unit.cpp")],
    }
    assert camera_violation.read_text(encoding="utf-8") == camera_contents
