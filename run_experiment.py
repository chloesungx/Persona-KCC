"""Run one KCC external-information condition on the bundled ANES sample."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.anes_utils import preprocess_anes_2024
from src.kcc.case_configs import CASE_CONFIGS
from src.kcc.persona_kcc import build_personas
from src.llm.factory import get_llm


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("A", "B", "C", "D")


def load_runner(condition: str):
    """Import only the selected condition and its optional dependencies."""
    if condition == "A":
        from src.kcc.simulation_A import load_prompt, simulate_event_A
        return load_prompt, simulate_event_A
    if condition == "B":
        from src.kcc.simulation_B import load_prompt, simulate_event_B
        return load_prompt, simulate_event_B
    if condition == "C":
        from src.kcc.simulation_C import load_prompt, simulate_event_C
        return load_prompt, simulate_event_C
    from src.kcc.simulation_D import load_prompt, simulate_event_D
    return load_prompt, simulate_event_D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--num-personas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "output")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = CASE_CONFIGS["Climate Change (VCF0842)"]
    options = [f'{item["code"]}. {item["label"]}' for item in config["options"]]

    frame = preprocess_anes_2024(
        str(ROOT / "data" / "anes_2020_2024_100.csv"),
        target_col=config["target_vcf"],
    )
    frame = frame.dropna(subset=[config["target_vcf"]])
    frame = frame.sample(min(args.num_personas, len(frame)), random_state=args.seed)
    personas = build_personas(
        frame,
        config["query"],
        val_mode=True,
        vcf_keys=config["survey_vcf_keys"],
    )

    load_prompt, runner = load_runner(args.condition)
    result = runner(
        llm=get_llm(args.model),
        personas=personas,
        prompt_templates=load_prompt("v15"),
        read_context_templates=load_prompt("v11_read_context"),
        query=config["query"],
        event=config["event"],
        options=options,
        val_mode=True,
        target_vcf=config["target_vcf"],
        data="balance",
        history=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"condition_{args.condition}_seed_{args.seed}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
