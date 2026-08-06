from pathlib import Path

import pandas as pd

from scripts.expand_wrds_cds_redcode_map import expand_map


def test_expand_map_adds_safe_alias_matches(tmp_path: Path) -> None:
    broad = pd.DataFrame(
        [
            {"ticker": "AMERAIAI", "shortname": "Amern Airls Group Inc", "redcode": "025ADX"},
            {"ticker": "KRAFHEI", "shortname": "Kraft Heinz Foods Co", "redcode": "5F07CP"},
            {"ticker": "OTHER", "shortname": "Other Name", "redcode": "XXXXXX"},
        ]
    )
    broad_path = tmp_path / "broad.csv.gz"
    broad.to_csv(broad_path, index=False, compression="gzip")

    partial = pd.DataFrame(
        [
            {
                "company_id": "0000000001",
                "company_name": "Existing Co",
                "equity_ticker": "EX",
                "cds_ticker": "EXIST",
                "redcode": "ABC123",
                "shortname": "Existing Co",
                "match_type": "exact_ticker",
            }
        ]
    )
    partial_path = tmp_path / "partial.csv"
    partial.to_csv(partial_path, index=False)

    unresolved = pd.DataFrame(
        [
            {"company_id": "6201", "company_name": "AMERICAN AIRLINES GROUP INC.", "equity_ticker": "AAL"},
            {"company_id": "1637459", "company_name": "Kraft Heinz Co", "equity_ticker": "KHC"},
            {"company_id": "9999999999", "company_name": "No Match Co", "equity_ticker": "NMC"},
        ]
    )
    unresolved_path = tmp_path / "unresolved.csv"
    unresolved.to_csv(unresolved_path, index=False)

    additions, expanded, remaining, summary = expand_map(broad_path, partial_path, unresolved_path)

    assert set(additions["equity_ticker"]) == {"AAL", "KHC"}
    assert set(additions["match_type"]) == {"token_exact_abbrev", "token_subset_one_extra"}
    assert len(expanded) == 3
    assert remaining["equity_ticker"].tolist() == ["NMC"]
    assert summary["new_safe_alias_rows"] == 2
