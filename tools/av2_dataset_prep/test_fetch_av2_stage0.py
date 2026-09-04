"""Unit tests for the read-only AV2 Stage-0 fetch tool.

Every test here is offline: the S3 XML parser is tested against a fixture
string, and the download function against an injected fake opener --
no real network access is exercised or required.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_av2_stage0 import (  # noqa: E402
    Av2FetchError,
    build_manifest,
    download_scenario_parquet,
    list_scenario_id_pool,
    parse_listing_xml,
    scenario_remote_key,
    select_stage0,
)

FIXTURE_LISTING_PAGE_1 = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<Name>argoverse</Name>
<Prefix>datasets/av2/motion-forecasting/train/</Prefix>
<KeyCount>3</KeyCount><MaxKeys>3</MaxKeys><Delimiter>/</Delimiter>
<IsTruncated>true</IsTruncated>
<NextContinuationToken>TOKEN-1</NextContinuationToken>
<CommonPrefixes><Prefix>datasets/av2/motion-forecasting/train/aaa-1/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>datasets/av2/motion-forecasting/train/aaa-2/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>datasets/av2/motion-forecasting/train/aaa-3/</Prefix></CommonPrefixes>
</ListBucketResult>"""

FIXTURE_LISTING_PAGE_2 = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<Name>argoverse</Name>
<Prefix>datasets/av2/motion-forecasting/train/</Prefix>
<KeyCount>2</KeyCount><MaxKeys>3</MaxKeys><Delimiter>/</Delimiter>
<IsTruncated>false</IsTruncated>
<CommonPrefixes><Prefix>datasets/av2/motion-forecasting/train/aaa-4/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>datasets/av2/motion-forecasting/train/aaa-5/</Prefix></CommonPrefixes>
</ListBucketResult>"""

FIXTURE_SCENARIO_CONTENTS_PAGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<Name>argoverse</Name>
<Prefix>datasets/av2/motion-forecasting/train/aaa-1/</Prefix>
<KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>
<Contents><Key>datasets/av2/motion-forecasting/train/aaa-1/log_map_archive_aaa-1.json</Key>
<LastModified>2023-03-24T20:40:14.000Z</LastModified><Size>54646</Size></Contents>
<Contents><Key>datasets/av2/motion-forecasting/train/aaa-1/scenario_aaa-1.parquet</Key>
<LastModified>2023-03-24T20:40:14.000Z</LastModified><Size>128102</Size></Contents>
</ListBucketResult>"""


def test_parse_listing_xml_extracts_common_prefixes_and_token():
    common_prefixes, keys, token = parse_listing_xml(FIXTURE_LISTING_PAGE_1)
    assert common_prefixes == [
        "datasets/av2/motion-forecasting/train/aaa-1/",
        "datasets/av2/motion-forecasting/train/aaa-2/",
        "datasets/av2/motion-forecasting/train/aaa-3/",
    ]
    assert keys == []
    assert token == "TOKEN-1"


def test_parse_listing_xml_extracts_object_keys():
    _common_prefixes, keys, token = parse_listing_xml(FIXTURE_SCENARIO_CONTENTS_PAGE)
    assert keys == [
        "datasets/av2/motion-forecasting/train/aaa-1/log_map_archive_aaa-1.json",
        "datasets/av2/motion-forecasting/train/aaa-1/scenario_aaa-1.parquet",
    ]
    assert token is None


def test_list_scenario_id_pool_paginates_and_extracts_ids():
    pages = [FIXTURE_LISTING_PAGE_1, FIXTURE_LISTING_PAGE_2]

    def fake_fetcher(prefix, token):
        assert prefix == "datasets/av2/motion-forecasting/train/"
        return pages.pop(0)

    pool = list_scenario_id_pool("train", pool_size=10, _page_fetcher=fake_fetcher)
    assert pool == ["aaa-1", "aaa-2", "aaa-3", "aaa-4", "aaa-5"]


def test_list_scenario_id_pool_respects_pool_size_cap():
    def fake_fetcher(prefix, token):
        return FIXTURE_LISTING_PAGE_1  # always truncated, would loop forever without a cap

    pool = list_scenario_id_pool("train", pool_size=2, _page_fetcher=fake_fetcher)
    assert pool == ["aaa-1", "aaa-2"]


def test_list_scenario_id_pool_rejects_unsupported_split():
    with pytest.raises(Av2FetchError):
        list_scenario_id_pool("bogus_split", pool_size=10)


def test_select_stage0_is_deterministic_given_seed():
    pool = [f"scenario-{i:04d}" for i in range(500)]
    a = select_stage0(pool, count=50, seed=42)
    b = select_stage0(pool, count=50, seed=42)
    assert a == b
    assert len(a) == 50
    assert len(set(a)) == 50  # no duplicates
    assert set(a).issubset(set(pool))


def test_select_stage0_different_seed_gives_different_selection():
    pool = [f"scenario-{i:04d}" for i in range(500)]
    a = select_stage0(pool, count=50, seed=1)
    b = select_stage0(pool, count=50, seed=2)
    assert a != b


def test_select_stage0_rejects_count_exceeding_pool():
    with pytest.raises(Av2FetchError):
        select_stage0(["a", "b"], count=5, seed=0)


def test_scenario_remote_key_matches_real_verified_layout():
    key = scenario_remote_key("train", "abc-123")
    assert key == "datasets/av2/motion-forecasting/train/abc-123/scenario_abc-123.parquet"
    assert "log_map_archive" not in key  # never the map file


def test_download_scenario_parquet_writes_file_and_hashes(tmp_path):
    payload = b"fake parquet bytes for testing"

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(payload)

        def __exit__(self, *exc):
            return False

    result = download_scenario_parquet(
        "train", "abc-123", tmp_path, _opener=lambda url: FakeResponse()
    )
    assert result.bytes_downloaded == len(payload)
    assert not result.already_present
    written = Path(result.local_path).read_bytes()
    assert written == payload
    import hashlib

    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_download_scenario_parquet_skips_already_present(tmp_path):
    local_dir = tmp_path / "train" / "abc-123"
    local_dir.mkdir(parents=True)
    (local_dir / "scenario_abc-123.parquet").write_bytes(b"already here")

    def opener_that_must_not_be_called(url):
        raise AssertionError("opener should not be called when file already present")

    result = download_scenario_parquet(
        "train", "abc-123", tmp_path, _opener=opener_that_must_not_be_called
    )
    assert result.already_present


def test_build_manifest_records_provenance_fields():
    class FakeResult:
        def __init__(self, scenario_id):
            self.scenario_id = scenario_id
            self.split = "train"
            self.remote_key = scenario_remote_key("train", scenario_id)
            self.local_path = f"/x/{scenario_id}.parquet"
            self.bytes_downloaded = 1234
            self.sha256 = "deadbeef"
            self.already_present = False

    manifest = build_manifest("train", pool_size=2000, seed=20260905, results=[FakeResult("s1"), FakeResult("s2")])
    assert manifest["split"] == "train"
    assert manifest["selection_seed"] == 20260905
    assert manifest["selection_pool_size"] == 2000
    assert manifest["scenario_count"] == 2
    assert manifest["total_bytes_downloaded"] == 2468
    assert manifest["scenarios"][0]["scenario_id"] == "s1"
    assert "log_map_archive" not in manifest["scenarios"][0]["remote_key"]
