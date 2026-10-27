"""Experiment settings used for the KCC environmental-regulation study."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
SURVEY_MAP_PATH = ROOT / "data" / "survey" / "survey_anes_VCF0842.json"

with SURVEY_MAP_PATH.open(encoding="utf-8") as file:
    SURVEY_MAP = json.load(file)

CASE_CONFIGS = {
    "Climate Change (VCF0842)": {
        "name": "Environmental regulation (VCF0842)",
        "event": (
            "There is an ongoing public debate about government regulation "
            "of business for environmental protection."
        ),
        "query": (
            "Where would you place yourself on a 1-to-7 scale, where 1 means "
            "tougher regulations to protect the environment and 7 means "
            "regulations are already too much of a burden on business?"
        ),
        "target_vcf": "VCF0842",
        "options": [{"code": value, "label": str(value)} for value in range(1, 8)],
        "survey_vcf_keys": list(SURVEY_MAP.keys()),
    }
}
