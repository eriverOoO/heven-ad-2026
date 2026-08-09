from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci_image_fingerprint.py"


def write_fixture(root: Path) -> None:
    (root / "docker" / "ci").mkdir(parents=True)
    (root / "patches" / "dependency").mkdir(parents=True)
    (root / "package_a").mkdir()
    (root / "package_b").mkdir()
    (root / "scripts").mkdir()
    (root / "docker" / "ci" / "Dockerfile").write_text("FROM scratch\n")
    (root / "docker" / "ci" / "rebuild-revision").write_text("1\n")
    (root / "dependencies.repos").write_text("repositories: {}\n")
    (root / "scripts" / "apply_dependency_patches.sh").write_text("exit 0\n")
    (root / "patches" / "dependency" / "fix.patch").write_text("patch\n")
    (root / "package_a" / "package.xml").write_text("<package>a</package>\n")
    (root / "package_b" / "package.xml").write_text("<package>b</package>\n")


def fingerprint(root: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_fingerprint_is_deterministic_and_dependency_sensitive(tmp_path):
    write_fixture(tmp_path)

    first = fingerprint(tmp_path)
    second = fingerprint(tmp_path)
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")

    (tmp_path / "package_b" / "package.xml").write_text("<package>b2</package>\n")
    assert fingerprint(tmp_path) != first


def test_fingerprint_ignores_application_source_but_honors_manual_revision(tmp_path):
    write_fixture(tmp_path)
    (tmp_path / "package_a" / "node.py").write_text("print('first')\n")
    first = fingerprint(tmp_path)

    (tmp_path / "package_a" / "node.py").write_text("print('second')\n")
    assert fingerprint(tmp_path) == first

    (tmp_path / "docker" / "ci" / "rebuild-revision").write_text("2\n")
    assert fingerprint(tmp_path) != first


def test_fingerprint_fails_when_required_inputs_are_missing(tmp_path):
    write_fixture(tmp_path)
    (tmp_path / "dependencies.repos").unlink()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing dependency fingerprint input" in result.stderr
