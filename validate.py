#!/usr/bin/env python3
"""
Gold Standard Drug Validation Script
====================================

Validates computational brain model against clinical pharmacological data.

Usage:
    python validate.py --drug propofol --dose 140 --route IV --output results/propofol_001.json

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation.network_builder import BrainNetwork, NetworkParameters, build_m1_network, build_rtx3050_network
from pharmacology.pharmacokinetics import (
    simulate_pk_profile,
    RouteOfAdministration,
    DrugPKDatabase
)


class DrugValidator:
    """Validates drug effects against clinical data."""

    # Clinical validation targets from literature
    VALIDATION_TARGETS = {
        "propofol": {
            "name": "Propofol",
            "mechanism": "GABA_A positive allosteric modulator",
            "dose_mg_kg": 2.0,
            "route": "IV",
            "eeg_suppression_pct": 60.0,
            "eeg_suppression_tolerance": 10.0,  # ±10%
            "onset_time_min": 0.75,  # 45 seconds
            "reference": "Brown EN et al. (2011) NEJM 363:2638",
        },
        "ketamine": {
            "name": "Ketamine",
            "mechanism": "NMDA receptor antagonist",
            "dose_mg_kg": 2.0,
            "route": "IM",
            "gamma_power_increase": 2.5,  # 2.5x increase (30-80 Hz)
            "gamma_tolerance": 0.5,
            "onset_time_min": 5.0,
            "reference": "Sleigh JW et al. (2014) Br J Anaesth 113:i61",
        },
        "levodopa": {
            "name": "Levodopa",
            "mechanism": "Dopamine precursor",
            "dose_mg": 100,
            "route": "oral",
            "updrs_improvement_pct": 40.0,  # 30-50% range
            "updrs_tolerance": 10.0,
            "onset_time_min": 60.0,  # 1 hour
            "reference": "Poewe W et al. (2017) Nat Rev Dis Primers 3:17013",
        },
        "fluoxetine": {
            "name": "Fluoxetine (Prozac)",
            "mechanism": "SSRI (SERT inhibitor)",
            "dose_mg": 20,
            "route": "oral",
            "serotonin_increase_nM": 50.0,  # 5x baseline (10 nM → 50 nM)
            "serotonin_tolerance": 20.0,
            "response_latency_weeks": 3.0,  # 2-4 weeks
            "reference": "Wong DT et al. (2005) Nat Rev Drug Discov 4:764",
        },
        "diazepam": {
            "name": "Diazepam (Valium)",
            "mechanism": "GABA_A benzodiazepine modulator",
            "dose_mg": 10,
            "route": "oral",
            "beta_power_increase_pct": 40.0,  # 13-30 Hz increase
            "beta_tolerance": 15.0,
            "onset_time_min": 30.0,
            "reference": "Olkkola KT, Ahonen J (2008) Clin Pharmacokinet 47:469",
        },
    }

    def __init__(self, hardware: str = "m1"):
        """
        Initialize validator.

        Args:
            hardware: 'm1' for MacBook Air or 'rtx3050' for HP Victus
        """
        self.hardware = hardware
        print(f"Initializing validator for {hardware.upper()} hardware...")

        # Build network
        if hardware == "m1":
            self.network = build_m1_network()
        else:
            self.network = build_rtx3050_network()

        print(f"Network initialized: {len(self.network.neurons):,} neurons")

    def validate_drug(
        self,
        drug_name: str,
        dose: float,
        route: str,
        duration_hours: float = 24.0,
        body_weight_kg: float = 70.0
    ) -> Dict:
        """
        Validate a drug against clinical data.

        Args:
            drug_name: Drug name ('propofol', 'ketamine', etc.)
            dose: Dose (mg or mg/kg depending on drug)
            route: Route of administration ('IV', 'oral', 'IM')
            duration_hours: Simulation duration
            body_weight_kg: Patient weight

        Returns:
            Validation results dictionary
        """
        if drug_name not in self.VALIDATION_TARGETS:
            raise ValueError(f"Unknown drug: {drug_name}. Must be one of {list(self.VALIDATION_TARGETS.keys())}")

        target = self.VALIDATION_TARGETS[drug_name]
        print(f"\n{'=' * 80}")
        print(f"Validating: {target['name']}")
        print(f"Mechanism: {target['mechanism']}")
        print(f"Reference: {target['reference']}")
        print(f"{'=' * 80}\n")

        # 1. Pharmacokinetics: Calculate brain concentration
        print("Step 1: Simulating pharmacokinetics...")

        # Convert dose if needed (mg/kg → mg)
        if "dose_mg_kg" in target:
            dose_mg = dose * body_weight_kg
        else:
            dose_mg = dose

        route_enum = getattr(RouteOfAdministration, route.upper())

        pk_result = simulate_pk_profile(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route_enum,
            duration_hours=duration_hours,
            body_weight_kg=body_weight_kg
        )

        # Find peak brain concentration
        peak_idx = np.argmax(pk_result["brain_concentration_uM"])
        peak_brain_conc_uM = pk_result["brain_concentration_uM"][peak_idx]
        peak_time_hours = pk_result["time_hours"][peak_idx]

        print(f"  Peak brain concentration: {peak_brain_conc_uM:.2f} μM at {peak_time_hours:.2f} h")

        # 2. Pharmacodynamics: Apply drug to network
        print("\nStep 2: Applying drug to brain network...")

        effects = self.network.apply_drug(
            drug_name=drug_name,
            concentration=peak_brain_conc_uM,
            unit="uM"
        )

        print(f"  Receptor targets: {', '.join(effects['receptor_targets'])}")

        # 3. Validation: Compare with clinical targets
        print("\nStep 3: Validating against clinical data...")

        validation_results = {
            "drug_name": drug_name,
            "dose_mg": dose_mg,
            "route": route,
            "timestamp": datetime.now().isoformat(),
            "hardware": self.hardware,
            "pk_profile": {
                "peak_brain_concentration_uM": float(peak_brain_conc_uM),
                "peak_time_hours": float(peak_time_hours),
            },
            "network_effects": effects["network_effects"],
            "clinical_targets": target,
            "validation_metrics": {},
        }

        # Calculate errors for each drug
        if drug_name == "propofol":
            sim_suppression = effects["network_effects"]["eeg_suppression_pct"]
            target_suppression = target["eeg_suppression_pct"]
            error_pct = abs(sim_suppression - target_suppression) / target_suppression * 100

            validation_results["validation_metrics"] = {
                "simulated_eeg_suppression_pct": float(sim_suppression),
                "target_eeg_suppression_pct": float(target_suppression),
                "error_pct": float(error_pct),
                "within_tolerance": bool(error_pct <= 15.0),  # Acceptance: <15% error
            }

            print(f"  Simulated EEG suppression: {sim_suppression:.1f}%")
            print(f"  Target: {target_suppression:.1f}% ± {target['eeg_suppression_tolerance']:.1f}%")
            print(f"  Error: {error_pct:.1f}%")
            print(f"  ✓ PASS" if validation_results["validation_metrics"]["within_tolerance"] else "  ✗ FAIL")

        elif drug_name == "ketamine":
            sim_gamma = effects["network_effects"]["gamma_power_increase"]
            target_gamma = target["gamma_power_increase"]
            error_pct = abs(sim_gamma - target_gamma) / target_gamma * 100

            validation_results["validation_metrics"] = {
                "simulated_gamma_increase": float(sim_gamma),
                "target_gamma_increase": float(target_gamma),
                "error_pct": float(error_pct),
                "within_tolerance": bool(error_pct <= 15.0),
            }

            print(f"  Simulated gamma increase: {sim_gamma:.2f}x")
            print(f"  Target: {target_gamma:.2f}x ± {target['gamma_tolerance']:.2f}x")
            print(f"  Error: {error_pct:.1f}%")
            print(f"  ✓ PASS" if validation_results["validation_metrics"]["within_tolerance"] else "  ✗ FAIL")

        elif drug_name == "levodopa":
            sim_improvement = effects["network_effects"]["motor_improvement_pct"]
            target_improvement = target["updrs_improvement_pct"]
            error_pct = abs(sim_improvement - target_improvement) / target_improvement * 100

            validation_results["validation_metrics"] = {
                "simulated_updrs_improvement_pct": float(sim_improvement),
                "target_updrs_improvement_pct": float(target_improvement),
                "error_pct": float(error_pct),
                "within_tolerance": bool(error_pct <= 15.0),
            }

            print(f"  Simulated UPDRS improvement: {sim_improvement:.1f}%")
            print(f"  Target: {target_improvement:.1f}% ± {target['updrs_tolerance']:.1f}%")
            print(f"  Error: {error_pct:.1f}%")
            print(f"  ✓ PASS" if validation_results["validation_metrics"]["within_tolerance"] else "  ✗ FAIL")

        elif drug_name == "fluoxetine":
            sim_serotonin = effects["network_effects"]["serotonin_increase_nM"]
            target_serotonin = target["serotonin_increase_nM"]
            error_pct = abs(sim_serotonin - target_serotonin) / target_serotonin * 100

            validation_results["validation_metrics"] = {
                "simulated_serotonin_nM": float(sim_serotonin),
                "target_serotonin_nM": float(target_serotonin),
                "error_pct": float(error_pct),
                "within_tolerance": bool(error_pct <= 15.0),
            }

            print(f"  Simulated serotonin: {sim_serotonin:.1f} nM")
            print(f"  Target: {target_serotonin:.1f} ± {target['serotonin_tolerance']:.1f} nM")
            print(f"  Error: {error_pct:.1f}%")
            print(f"  ✓ PASS" if validation_results["validation_metrics"]["within_tolerance"] else "  ✗ FAIL")

        elif drug_name == "diazepam":
            sim_beta = effects["network_effects"]["beta_power_increase_pct"]
            target_beta = target["beta_power_increase_pct"]
            error_pct = abs(sim_beta - target_beta) / target_beta * 100

            validation_results["validation_metrics"] = {
                "simulated_beta_increase_pct": float(sim_beta),
                "target_beta_increase_pct": float(target_beta),
                "error_pct": float(error_pct),
                "within_tolerance": bool(error_pct <= 15.0),
            }

            print(f"  Simulated beta increase: {sim_beta:.1f}%")
            print(f"  Target: {target_beta:.1f}% ± {target['beta_tolerance']:.1f}%")
            print(f"  Error: {error_pct:.1f}%")
            print(f"  ✓ PASS" if validation_results["validation_metrics"]["within_tolerance"] else "  ✗ FAIL")

        return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate drug effects in brain simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --drug propofol --dose 2.0 --route IV
  python validate.py --drug ketamine --dose 2.0 --route IM
  python validate.py --drug levodopa --dose 100 --route oral
  python validate.py --drug fluoxetine --dose 20 --route oral
  python validate.py --drug diazepam --dose 10 --route oral
        """
    )

    parser.add_argument(
        "--drug",
        type=str,
        required=True,
        choices=["propofol", "ketamine", "levodopa", "fluoxetine", "diazepam"],
        help="Drug to validate"
    )

    parser.add_argument(
        "--dose",
        type=float,
        required=True,
        help="Dose (mg/kg for propofol/ketamine, mg for others)"
    )

    parser.add_argument(
        "--route",
        type=str,
        default="IV",
        choices=["IV", "oral", "IM", "SC"],
        help="Route of administration"
    )

    parser.add_argument(
        "--hardware",
        type=str,
        default="m1",
        choices=["m1", "rtx3050"],
        help="Hardware to use (m1 or rtx3050)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/<drug>_<timestamp>.json)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=24.0,
        help="Simulation duration in hours (default: 24)"
    )

    args = parser.parse_args()

    # Create validator
    validator = DrugValidator(hardware=args.hardware)

    # Run validation
    results = validator.validate_drug(
        drug_name=args.drug,
        dose=args.dose,
        route=args.route,
        duration_hours=args.duration
    )

    # Save results
    if args.output is None:
        output_dir = Path("results") / "goldstandard"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.drug}_{timestamp}.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")

    # Exit with status code
    if results["validation_metrics"]["within_tolerance"]:
        print("✓ VALIDATION PASSED\n")
        sys.exit(0)
    else:
        print("✗ VALIDATION FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
