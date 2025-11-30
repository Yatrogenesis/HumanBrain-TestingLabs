#!/usr/bin/env python3
"""
Quick Gold Standard Validation (No full network)
================================================

Validates pharmacological models using receptor dynamics directly.
This is faster than building the full 100K neuron network.

Author: Francisco Molina Burgos (Yatrogenesis)
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation.synapse_models import GABAaReceptor, NMDAReceptor, DopamineD2Receptor, SerotoninTransporter, MuOpioidReceptor, DopamineD2Antagonist
from pharmacology.pharmacokinetics import simulate_pk_profile, RouteOfAdministration

# Clinical validation targets
TARGETS = {
    "propofol": {"target": 60.0, "tolerance": 15.0, "metric": "eeg_suppression_pct"},
    "ketamine": {"target": 2.5, "tolerance": 15.0, "metric": "gamma_increase"},
    "levodopa": {"target": 40.0, "tolerance": 15.0, "metric": "motor_improvement_pct"},
    "fluoxetine": {"target": 50.0, "tolerance": 30.0, "metric": "serotonin_nM"},
    "diazepam": {"target": 40.0, "tolerance": 15.0, "metric": "beta_increase_pct"},
    "morphine": {"target": 50.0, "tolerance": 15.0, "metric": "analgesia_pct"},
    "haloperidol": {"target": 65.0, "tolerance": 15.0, "metric": "d2_occupancy_pct"},
    "midazolam": {"target": 70.0, "tolerance": 15.0, "metric": "sedation_pct"},
}

def validate_propofol(dose_mg_kg=2.0, body_weight=70.0):
    """Validate propofol against 60% EEG suppression target."""
    dose_mg = dose_mg_kg * body_weight

    # PK: Get brain concentration
    pk = simulate_pk_profile("propofol", dose_mg, RouteOfAdministration.IV, duration_hours=4.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    # PD: GABA_A receptor effect
    receptor = GABAaReceptor()
    receptor.bind_drug(peak_brain_uM, drug_type="propofol", efficacy=0.85)
    suppression = receptor.get_suppression_percentage()

    target = TARGETS["propofol"]["target"]
    error = abs(suppression - target) / target * 100
    passed = error <= TARGETS["propofol"]["tolerance"]

    return {
        "drug": "propofol",
        "peak_brain_uM": float(peak_brain_uM),
        "simulated": float(suppression),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_ketamine(dose_mg_kg=2.0, body_weight=70.0):
    """Validate ketamine against 2.5x gamma power increase target."""
    dose_mg = dose_mg_kg * body_weight

    pk = simulate_pk_profile("ketamine", dose_mg, RouteOfAdministration.IM, duration_hours=8.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    receptor = NMDAReceptor()
    receptor.bind_ketamine(peak_brain_uM)
    gamma_increase = receptor.get_gamma_power_increase()

    target = TARGETS["ketamine"]["target"]
    error = abs(gamma_increase - target) / target * 100
    passed = error <= TARGETS["ketamine"]["tolerance"]

    return {
        "drug": "ketamine",
        "peak_brain_uM": float(peak_brain_uM),
        "simulated": float(gamma_increase),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_levodopa(dose_mg=100.0, body_weight=70.0):
    """Validate levodopa against 40% UPDRS improvement target."""
    pk = simulate_pk_profile("levodopa", dose_mg, RouteOfAdministration.ORAL, duration_hours=8.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    # Use AADC enzyme to convert L-DOPA to dopamine (realistic mechanism)
    receptor = DopamineD2Receptor()
    improvement = receptor.convert_ldopa_to_motor_effect(peak_brain_uM)

    target = TARGETS["levodopa"]["target"]
    error = abs(improvement - target) / target * 100
    passed = error <= TARGETS["levodopa"]["tolerance"]

    return {
        "drug": "levodopa",
        "peak_brain_uM": float(peak_brain_uM),
        "simulated": float(improvement),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_fluoxetine(dose_mg=20.0, body_weight=70.0):
    """Validate fluoxetine against 50 nM serotonin target."""
    pk = simulate_pk_profile("fluoxetine", dose_mg, RouteOfAdministration.ORAL, duration_hours=168.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])
    peak_brain_nM = peak_brain_uM * 1000

    transporter = SerotoninTransporter()
    transporter.bind_fluoxetine(peak_brain_nM)
    serotonin = transporter.calculate_synaptic_serotonin()

    target = TARGETS["fluoxetine"]["target"]
    error = abs(serotonin - target) / target * 100
    passed = error <= TARGETS["fluoxetine"]["tolerance"]

    return {
        "drug": "fluoxetine",
        "peak_brain_nM": float(peak_brain_nM),
        "simulated": float(serotonin),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_diazepam(dose_mg=10.0, body_weight=70.0):
    """Validate diazepam against 40% beta power increase target."""
    pk = simulate_pk_profile("diazepam", dose_mg, RouteOfAdministration.ORAL, duration_hours=72.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    receptor = GABAaReceptor()
    receptor.bind_drug(peak_brain_uM, drug_type="diazepam")  # Uses BZ-specific parameters
    beta_increase = receptor.get_beta_power_increase_percent()

    target = TARGETS["diazepam"]["target"]
    error = abs(beta_increase - target) / target * 100
    passed = error <= TARGETS["diazepam"]["tolerance"]

    return {
        "drug": "diazepam",
        "peak_brain_uM": float(peak_brain_uM),
        "simulated": float(beta_increase),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }


def validate_morphine(dose_mg=10.0, body_weight=70.0):
    """Validate morphine against 50% pain reduction target."""
    pk = simulate_pk_profile("morphine", dose_mg, RouteOfAdministration.IV, duration_hours=6.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])
    peak_brain_nM = peak_brain_uM * 1000

    receptor = MuOpioidReceptor()
    receptor.bind_morphine(peak_brain_nM)
    analgesia = receptor.get_analgesia_percent()

    target = TARGETS["morphine"]["target"]
    error = abs(analgesia - target) / target * 100
    passed = error <= TARGETS["morphine"]["tolerance"]

    return {
        "drug": "morphine",
        "peak_brain_nM": float(peak_brain_nM),
        "simulated": float(analgesia),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_haloperidol(dose_mg=5.0, body_weight=70.0):
    """Validate haloperidol against 65% D2 occupancy target."""
    pk = simulate_pk_profile("haloperidol", dose_mg, RouteOfAdministration.ORAL, duration_hours=24.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])
    peak_brain_nM = peak_brain_uM * 1000

    receptor = DopamineD2Antagonist()
    receptor.bind_haloperidol(peak_brain_nM)
    d2_occupancy = receptor.get_d2_occupancy_percent()

    target = TARGETS["haloperidol"]["target"]
    error = abs(d2_occupancy - target) / target * 100
    passed = error <= TARGETS["haloperidol"]["tolerance"]

    return {
        "drug": "haloperidol",
        "peak_brain_nM": float(peak_brain_nM),
        "simulated": float(d2_occupancy),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }

def validate_midazolam(dose_mg_kg=0.1, body_weight=70.0):
    """Validate midazolam against 70% sedation target (Ramsay 3-4)."""
    dose_mg = dose_mg_kg * body_weight
    pk = simulate_pk_profile("midazolam", dose_mg, RouteOfAdministration.IV, duration_hours=4.0)
    peak_brain_uM = max(pk["brain_concentration_uM"])

    receptor = GABAaReceptor()
    receptor.bind_drug(peak_brain_uM, drug_type="midazolam")
    sedation = receptor.get_sedation_percentage()  # Uses Emax model for ARAS inhibition

    target = TARGETS["midazolam"]["target"]
    error = abs(sedation - target) / target * 100
    passed = error <= TARGETS["midazolam"]["tolerance"]

    return {
        "drug": "midazolam",
        "peak_brain_uM": float(peak_brain_uM),
        "simulated": float(sedation),
        "target": float(target),
        "error_pct": float(error),
        "passed": bool(passed)
    }


def main():
    print("\n" + "=" * 80)
    print("GOLD STANDARD QUICK VALIDATION")
    print("=" * 80 + "\n")

    validators = [
        validate_propofol,
        validate_ketamine,
        validate_levodopa,
        validate_fluoxetine,
        validate_diazepam,
        validate_morphine,
        validate_haloperidol,
        validate_midazolam,
    ]

    results = []
    for validator in validators:
        result = validator()
        results.append(result)

        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{result['drug'].upper():12} | Simulated: {result['simulated']:6.1f} | Target: {result['target']:6.1f} | Error: {result['error_pct']:5.1f}% | {status}")

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("=" * 80 + "\n")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {"passed": passed, "total": total, "pass_rate": passed/total*100}
    }

    output_file = Path("results/quick_validation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
