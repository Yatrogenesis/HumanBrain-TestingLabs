#!/usr/bin/env python3
"""
Validate All Gold Standard Drugs
=================================

Runs complete validation suite for all 5 gold standard drugs.

Usage:
    python validate_all_goldstandard.py --hardware m1 --replicates 3

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Drug protocols (clinical standard doses)
GOLD_STANDARD_DRUGS = [
    {
        "drug": "propofol",
        "dose": 2.0,  # mg/kg
        "route": "IV",
        "description": "GABA_A agonist anesthetic",
    },
    {
        "drug": "ketamine",
        "dose": 2.0,  # mg/kg
        "route": "IM",
        "description": "NMDA antagonist anesthetic",
    },
    {
        "drug": "levodopa",
        "dose": 100,  # mg
        "route": "oral",
        "description": "Dopamine precursor for Parkinson's",
    },
    {
        "drug": "fluoxetine",
        "dose": 20,  # mg
        "route": "oral",
        "description": "SSRI antidepressant",
    },
    {
        "drug": "diazepam",
        "dose": 10,  # mg
        "route": "oral",
        "description": "Benzodiazepine anxiolytic",
    },
]


def run_validation_suite(hardware: str = "m1", replicates: int = 1) -> Dict:
    """
    Run validation for all gold standard drugs.

    Args:
        hardware: 'm1' or 'rtx3050'
        replicates: Number of replicate runs per drug

    Returns:
        Summary of all validations
    """
    print(f"\n{'=' * 80}")
    print(f"GOLD STANDARD DRUG VALIDATION SUITE")
    print(f"Hardware: {hardware.upper()}")
    print(f"Replicates per drug: {replicates}")
    print(f"Total validations: {len(GOLD_STANDARD_DRUGS) * replicates}")
    print(f"{'=' * 80}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "goldstandard" / f"suite_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hardware,
        "replicates": replicates,
        "drugs_tested": [],
        "results": [],
        "aggregate_metrics": {},
    }

    all_errors = []
    all_passes = []

    # Run each drug
    for drug_config in GOLD_STANDARD_DRUGS:
        print(f"\n{'-' * 80}")
        print(f"Testing: {drug_config['drug'].upper()} - {drug_config['description']}")
        print(f"{'-' * 80}\n")

        summary["drugs_tested"].append(drug_config["drug"])

        drug_errors = []

        # Run replicates
        for rep in range(replicates):
            print(f"  Replicate {rep + 1}/{replicates}...")

            # Build command
            cmd = [
                sys.executable,
                "validate.py",
                "--drug", drug_config["drug"],
                "--dose", str(drug_config["dose"]),
                "--route", drug_config["route"],
                "--hardware", hardware,
                "--output", str(output_dir / f"{drug_config['drug']}_rep{rep + 1}.json"),
            ]

            # Run validation
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600  # 10 minute timeout
                )

                # Parse result file
                result_file = output_dir / f"{drug_config['drug']}_rep{rep + 1}.json"
                with open(result_file, 'r') as f:
                    validation_result = json.load(f)

                summary["results"].append(validation_result)

                # Extract error
                error_pct = validation_result["validation_metrics"]["error_pct"]
                passed = validation_result["validation_metrics"]["within_tolerance"]

                drug_errors.append(error_pct)
                all_errors.append(error_pct)
                all_passes.append(passed)

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"    Error: {error_pct:.2f}% - {status}")

            except subprocess.TimeoutExpired:
                print(f"    ✗ TIMEOUT (>10 min)")
            except Exception as e:
                print(f"    ✗ ERROR: {e}")

        # Drug summary
        if drug_errors:
            mean_error = sum(drug_errors) / len(drug_errors)
            print(f"\n  {drug_config['drug'].upper()} Summary:")
            print(f"    Mean error: {mean_error:.2f}%")
            print(f"    Min error: {min(drug_errors):.2f}%")
            print(f"    Max error: {max(drug_errors):.2f}%")

    # Aggregate metrics
    if all_errors:
        summary["aggregate_metrics"] = {
            "mean_error_pct": sum(all_errors) / len(all_errors),
            "median_error_pct": sorted(all_errors)[len(all_errors) // 2],
            "max_error_pct": max(all_errors),
            "min_error_pct": min(all_errors),
            "pass_rate": sum(all_passes) / len(all_passes) * 100,
            "total_validations": len(all_errors),
            "passed": sum(all_passes),
            "failed": len(all_passes) - sum(all_passes),
        }

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'=' * 80}")
    print(f"VALIDATION SUITE COMPLETE")
    print(f"{'=' * 80}\n")

    if summary["aggregate_metrics"]:
        metrics = summary["aggregate_metrics"]
        print(f"Total validations: {metrics['total_validations']}")
        print(f"Passed: {metrics['passed']} ({metrics['pass_rate']:.1f}%)")
        print(f"Failed: {metrics['failed']}")
        print(f"\nError Statistics:")
        print(f"  Mean: {metrics['mean_error_pct']:.2f}%")
        print(f"  Median: {metrics['median_error_pct']:.2f}%")
        print(f"  Range: {metrics['min_error_pct']:.2f}% - {metrics['max_error_pct']:.2f}%")

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run complete gold standard validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--hardware",
        type=str,
        default="m1",
        choices=["m1", "rtx3050"],
        help="Hardware to use (m1 or rtx3050)"
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of replicate runs per drug (default: 1)"
    )

    args = parser.parse_args()

    # Run suite
    summary = run_validation_suite(
        hardware=args.hardware,
        replicates=args.replicates
    )

    # Exit with appropriate code
    if summary["aggregate_metrics"]:
        if summary["aggregate_metrics"]["pass_rate"] == 100.0:
            print("✓ ALL VALIDATIONS PASSED\n")
            sys.exit(0)
        else:
            print("⚠ SOME VALIDATIONS FAILED\n")
            sys.exit(1)
    else:
        print("✗ NO RESULTS\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
