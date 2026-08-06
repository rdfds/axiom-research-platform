import json
from pathlib import Path
import zipfile

from src.sec_companyfacts_bulk import CompanyFactsBulkSource


def test_loads_payload_from_companyfacts_zip_and_hydrates_cache(tmp_path: Path) -> None:
    zip_path = tmp_path / "companyfacts.zip"
    payload = {"cik": "0000123456", "facts": {"us-gaap": {"Cash": {"units": {"USD": []}}}}}

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("companyfacts/CIK0000123456.json", json.dumps(payload))

    cache_dir = tmp_path / "cache"
    with CompanyFactsBulkSource(
        companyfacts_dir=cache_dir,
        companyfacts_zip=zip_path,
        hydrate_cache=True,
    ) as source:
        loaded = source.load("123456")

    assert loaded == payload
    assert json.loads((cache_dir / "CIK0000123456.json").read_text()) == payload


def test_prefers_local_cache_before_zip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_payload = {"cik": "0000123456", "source": "cache"}
    (cache_dir / "CIK0000123456.json").write_text(json.dumps(cached_payload))

    zip_path = tmp_path / "companyfacts.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CIK0000123456.json", json.dumps({"cik": "0000123456", "source": "zip"}))

    with CompanyFactsBulkSource(companyfacts_dir=cache_dir, companyfacts_zip=zip_path) as source:
        loaded = source.load("0000123456")

    assert loaded == cached_payload


def test_load_with_metadata_returns_raw_hash_and_origin(tmp_path: Path) -> None:
    zip_path = tmp_path / "companyfacts.zip"
    raw_payload = b'{"cik":"0000123456","source":"zip"}'

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CIK0000123456.json", raw_payload)

    with CompanyFactsBulkSource(companyfacts_zip=zip_path) as source:
        payload, raw_hash, origin = source.load_with_metadata("0000123456")

    assert payload == {"cik": "0000123456", "source": "zip"}
    assert raw_hash is not None
    assert origin == "zip"


def test_prefers_zip_when_requested(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "CIK0000123456.json").write_text(json.dumps({"cik": "0000123456", "source": "cache"}))

    zip_path = tmp_path / "companyfacts.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CIK0000123456.json", json.dumps({"cik": "0000123456", "source": "zip"}))

    with CompanyFactsBulkSource(
        companyfacts_dir=cache_dir,
        companyfacts_zip=zip_path,
        prefer_zip=True,
    ) as source:
        payload, _, origin = source.load_with_metadata("0000123456")

    assert payload == {"cik": "0000123456", "source": "zip"}
    assert origin == "zip"
