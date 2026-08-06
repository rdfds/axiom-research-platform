from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import zipfile


def normalize_cik(value: str) -> str:
    return str(value).strip().zfill(10)


class CompanyFactsBulkSource:
    """Fast local loader for SEC companyfacts payloads.

    It can read from a hydrated directory, a SEC `companyfacts.zip` archive,
    or both. When a zip is provided, requested payloads can optionally be
    hydrated into the local cache directory so later runs become filesystem-only.
    """

    def __init__(
        self,
        *,
        companyfacts_dir: Optional[Path] = None,
        companyfacts_zip: Optional[Path] = None,
        hydrate_cache: bool = False,
        prefer_zip: bool = False,
    ) -> None:
        self.companyfacts_dir = companyfacts_dir
        self.companyfacts_zip = companyfacts_zip
        self.hydrate_cache = hydrate_cache
        self.prefer_zip = prefer_zip
        self._zip_file: Optional[zipfile.ZipFile] = None
        self._zip_index: Optional[Dict[str, str]] = None

        if self.companyfacts_zip is not None:
            self._zip_file = zipfile.ZipFile(self.companyfacts_zip)
            self._zip_index = {}
            for member in self._zip_file.namelist():
                if not member.endswith(".json"):
                    continue
                base_name = Path(member).name
                if base_name.startswith("CIK") and base_name.endswith(".json"):
                    self._zip_index[base_name] = member

    def close(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None

    def __enter__(self) -> "CompanyFactsBulkSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _cached_path(self, cik: str) -> Optional[Path]:
        if self.companyfacts_dir is None:
            return None
        return self.companyfacts_dir / f"CIK{normalize_cik(cik)}.json"

    def load_with_metadata(self, cik: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        cik = normalize_cik(cik)
        cache_path = self._cached_path(cik)

        def read_cache() -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
            if cache_path is None or not cache_path.exists():
                return None, None, None
            raw_bytes = cache_path.read_bytes()
            return (
                json.loads(raw_bytes.decode("utf-8")),
                hashlib.sha256(raw_bytes).hexdigest(),
                "cache",
            )

        def read_zip() -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
            if self._zip_file is None or self._zip_index is None:
                return None, None, None
            member_name = self._zip_index.get(f"CIK{cik}.json")
            if member_name is None:
                return None, None, None

            raw_bytes = self._zip_file.read(member_name)
            payload = json.loads(raw_bytes.decode("utf-8"))
            if self.hydrate_cache and cache_path is not None and not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload))
            return payload, hashlib.sha256(raw_bytes).hexdigest(), "zip"

        readers = (read_zip, read_cache) if self.prefer_zip else (read_cache, read_zip)
        for reader in readers:
            payload, raw_hash, origin = reader()
            if payload is not None:
                return payload, raw_hash, origin
        return None, None, None

    def load(self, cik: str) -> Optional[Dict]:
        payload, _, _ = self.load_with_metadata(cik)
        return payload
