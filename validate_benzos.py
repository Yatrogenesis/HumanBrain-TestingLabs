#!/usr/bin/env python3
"""
Benzodiazepine Validation: Alprazolam & Clonazepam
==================================================

Validates that alprazolam and clonazepam produce similar beta power
increase (~40%) as diazepam, since they all act at the BZ site.

Author: Francisco Molina Burgos (Yatrogenesis)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation.synapse_models import GABAaReceptor
from pharmacology.pharmacokinetics import simulate_pk_profile, RouteOfAdministration

# Clinical targets for benzodiazepines
# All BZs should produce ~40% beta power increase at anxiolytic doses
TARGETS = {
    "alprazolam": {"target": 40.0, "tolerance": 15.0, "dose_mg": 0.5},
    "clonazepam": {"target": 40.0, "tolerance": 15.0, "dose_mg": 1.0},
    "diazepam": {"target": 40.0, "tolerance": 15.0, "dose_mg": 10.0},  # Reference
}


def validate_benzodiazepine(drug_name: str, dose_mg: float = None, body_weight: float = 70.0):
    """Validate a benzodiazepine against beta power increase target."""
    if dose_mg is None:
        dose_mg = TARGETS[drug_name]["dose_mg"]

    # Get duration based on half-life
    duration_map = {
        "alprazolam": 24.0,   # Short-acting
        "clonazepam": 72.0,   # Long-acting
        "diazepam": 72.0,     # Long-acting
    }
    duration = duration_map.get(drug_name, 48.0)

    # PK simulation
    pk = simulate_pk_profile(drug_name, dose_mg, RouteOfAdministration.ORAL, duration_hours=duration)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    # PD: GABA_A receptor effect at BZ site
    receptor = GABAaReceptor()
    receptor.bind_drug(peak_brain_uM, drug_type=drug_name, efficacy=0.70)  # BZ efficacy
    beta_increase = 100.0 * (receptor.modulation_factor - 1.0)

    target = TARGETS[drug_name]["target"]
    error = abs(beta_increase - target) / target * 100
    passed = error <= TARGETS[drug_name]["tolerance"]

    return {
        "drug": drug_name,
        "dose_mg": float(dose_mg),
        "peak_brain_uM": float(peak_brain_uM),
        "bound_fraction": float(receptor.drug_bound_fraction),
        "modulation_factor": float(receptor.modulation_factor),
        "beta_increase_pct": float(beta_increase),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }


def main():
    print("\n" + "=" * 80)
    print("BENZODIAZEPINE VALIDATION: Alprazolam & Clonazepam")
    print("=" * 80 + "\n")

    # Validate all three benzos for comparison
    drugs = ["diazepam", "alprazolam", "clonazepam"]
    results = []

    for drug in drugs:
        result = validate_benzodiazepine(drug)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"{drug.upper():12} | Dose: {result['dose_mg']:5.1f} mg | "
              f"Peak: {result['peak_brain_uM']:.3f} uM | "
              f"Bound: {result['bound_fraction']:.1%} | "
              f"Beta: {result['beta_increase_pct']:.1f}% | "
              f"Target: {result['target']:.0f}% | "
              f"Error: {result['error_pct']:.1f}% | {status}")

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} benzos passed ({passed/total*100:.0f}%)")
    print("=" * 80)

    # Detailed analysis
    print("\n=== PHARMACOLOGICAL ANALYSIS ===")
    print("\nBinding affinity comparison (lower IC50 = higher affinity):")
    print("  - Diazepam:   IC50 ~ 20 nM (reference)")
    print("  - Alprazolam: IC50 ~ 5 nM  (4x higher affinity)")
    print("  - Clonazepam: IC50 ~ 2 nM  (10x higher affinity)")

    print("\nExpected behavior:")
    print("  - Higher affinity BZs reach maximal binding at lower doses")
    print("  - All BZs produce similar max effect (~40% beta increase)")
    print("  - Different doses needed to achieve equivalent effect")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
