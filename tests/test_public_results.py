from __future__ import annotations

from scripts.verify_public_results import main


def test_public_result_claims_are_internally_consistent() -> None:
    assert main() == 0
