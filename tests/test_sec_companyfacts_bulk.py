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


