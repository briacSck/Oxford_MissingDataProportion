"""scripts/update_data_paths.py
--------------------------------
One-time script: writes source_data_file into spec.json for all papers.
Also fixes known spec errors (SMJ3560 column names).

Run from repo root:
    python scripts/update_data_paths.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Repo root (parent of this script's parent) ────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Data file mapping (relative paths from REPO_ROOT) ────────────────────────
DATA_FILE_MAP = {
    "Paper_AntiDiscrimination": "RA Missing data task/replication_pakacge/ad_dataset_ms1.dta",
    "Paper_Christensen2021":    "RA Missing data task/christensen-et-al-2021-hedging-on-the-hill-does-political-hedging-reduce-firm-risk/firmcyclepanel.dta",
    "Paper_Fang2022":           "RA Missing data task/fang-et-al-2022-anticorruption-government-subsidies-and-innovation-evidence-from-china/main_dataset.csv",
    "Paper_Hu2025":             "RA Missing data task/hu-et-al-2025-reshaping-corporate-boards-through-mandatory-gender-diversity-disclosures-evidence-from-canada/dirfin_main.dta",
    "Paper_IPOFirms":           "RA Missing data task/The Effect of Significant IPO Firms on Industry/sample_full.xlsx",
    "Paper_Mapping2024":        "RA Missing data task/0005 - Mapping entrepreneurial inclusion across US neighborhoods_ The case of low-code e-commerce entrepreneurship/mapping_entrepreneurial_inclusion.dta",
    "Paper_Meyer2024":          "RA Missing data task/SMJ 2024/SMJ_Final.dta",
    "Paper_P2P2018":            "RA Missing data task/P2P2018/P2P_dataset.dta",
    "Paper_SMJ3560":            "RA Missing data task/SMJ 3560/STATA CODE 3560.dta",
    "Paper_StatusConsensus":    "RA Missing data task/0017/movie_data.csv",
}

# ── Spec patches (applied on top of existing spec) ────────────────────────────
SPEC_PATCHES: dict[str, dict] = {
    "Paper_SMJ3560": {
        "key_independent_vars": ["Treatment"],   # was ["Treat"] — col doesn't exist in raw
        "control_vars": [
            "Age", "Registered", "Working", "Studying",
            "EntrepExperience", "RevenueLikert", "CustomersLikert",
        ],
        # Remove M1, R1, EDU3 — DO-file-generated dummies absent from raw data
        "parse_confidence": "high",
        "manual_review_required": False,
    }
}


def main() -> None:
    papers_dir = REPO_ROOT / "papers"
    updated = []
    skipped = []
    missing_data = []

    for paper_id, rel_path in DATA_FILE_MAP.items():
        spec_path = papers_dir / paper_id / "spec.json"
        if not spec_path.exists():
            skipped.append(f"{paper_id}: spec.json not found")
            continue

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        abs_path = REPO_ROOT / rel_path
        spec["source_data_file"] = str(abs_path)

        if not abs_path.exists():
            missing_data.append(f"{paper_id}: data file not found at {abs_path}")

        if paper_id in SPEC_PATCHES:
            spec.update(SPEC_PATCHES[paper_id])
            print(f"  [PATCH] {paper_id}: applied spec patches")

        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        updated.append(paper_id)
        print(f"  [OK]    {paper_id}: source_data_file set to {abs_path}")

    print(f"\nUpdated {len(updated)} spec files.")
    if skipped:
        print(f"Skipped  {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")
    if missing_data:
        print(f"\nWARNING — data files not on disk ({len(missing_data)}):")
        for s in missing_data:
            print(f"  {s}")


if __name__ == "__main__":
    main()
