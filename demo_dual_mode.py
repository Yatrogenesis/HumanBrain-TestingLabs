#!/usr/bin/env python3
"""
Demo: Dual-Mode Receptor System
===============================

Demonstrates:
1. DATABASE mode - Known drug simulation
2. MECHANISTIC mode - Novel drug prediction
3. REVERSE ENGINEERING - Infer drug from observed effect
4. DRUG INTERACTIONS - Synergy/competition analysis

Author: Francisco Molina Burgos (Yatrogenesis)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pharmacology.receptor_mechanisms import (
    UnifiedGABAaModel, ModelMode, BindingSite,
    EffectReverseEngineer, DRUG_PROFILES
)


def demo_database_mode():
    """Demo DATABASE mode with known drugs."""
    print("\n" + "=" * 80)
    print("1. DATABASE MODE - Known Drug Simulation")
    print("=" * 80)

    model = UnifiedGABAaModel(mode=ModelMode.DATABASE)

    # Test known drugs
    tests = [
        ("diazepam", 0.5),
        ("propofol", 5.0),
        ("alprazolam", 0.1),
    ]

    for drug, conc in tests:
        result = model.simulate(drug_name=drug, concentration_uM=conc)
        print(f"\n{drug.upper()} at {conc} uM:")
        print(f"  Mode: {result['mode']}")
        print(f"  Binding site: {result['binding_site']}")
        print(f"  Occupancy: {result['occupancy']:.1%}")
        print(f"  Modulation: {result['modulation']:.2f}x")
        print(f"  Beta increase: {result['beta_increase_pct']:.1f}%")
        print(f"  Sedation: {result['sedation_pct']:.1f}%")


def demo_mechanistic_mode():
    """Demo MECHANISTIC mode for novel compounds."""
    print("\n" + "=" * 80)
    print("2. MECHANISTIC MODE - Novel Drug Prediction")
    print("=" * 80)

    model = UnifiedGABAaModel(mode=ModelMode.MECHANISTIC)

    # Predict effect of hypothetical novel compounds
    novel_drugs = [
        {"name": "Novel BZ (high affinity)", "ki_nM": 1.0, "efficacy": 0.6, "site": BindingSite.BZ_SITE},
        {"name": "Novel Anesthetic (moderate)", "ki_nM": 2000, "efficacy": 0.7, "site": BindingSite.ANESTHETIC_SITE},
        {"name": "Novel Neurosteroid", "ki_nM": 500, "efficacy": 0.5, "site": BindingSite.NEUROSTEROID_SITE},
    ]

    for drug in novel_drugs:
        result = model.simulate(
            concentration_uM=0.5,
            ki_nM=drug["ki_nM"],
            efficacy=drug["efficacy"],
            binding_site=drug["site"]
        )
        print(f"\n{drug['name']}:")
        print(f"  Mode: {result['mode']}")
        print(f"  Ki: {drug['ki_nM']} nM")
        print(f"  Efficacy: {drug['efficacy']}")
        print(f"  Binding site: {drug['site'].value}")
        print(f"  At 0.5 uM:")
        print(f"    Occupancy: {result['occupancy']:.1%}")
        print(f"    Modulation: {result['modulation']:.2f}x")
        print(f"    Beta increase: {result['beta_increase_pct']:.1f}%")
        print(f"    Sedation: {result['sedation_pct']:.1f}%")


def demo_reverse_engineering():
    """Demo reverse engineering - infer drug from effect."""
    print("\n" + "=" * 80)
    print("3. REVERSE ENGINEERING - Infer Drug from Observed Effect")
    print("=" * 80)

    engineer = EffectReverseEngineer()

    # Scenario 1: Patient shows 60% sedation
    print("\n--- Scenario: Patient shows 60% sedation ---")
    candidates = engineer.infer_from_sedation(sedation_pct=60.0, tolerance=15.0)
    print(f"Candidate drugs that could cause 60% sedation:")
    for c in candidates[:5]:
        print(f"  - {c['drug']}: ~{c['estimated_concentration_uM']:.2f} uM ({c['clinical_plausibility']} plausibility)")

    # Scenario 2: EEG shows 40% beta increase
    print("\n--- Scenario: EEG shows 40% beta power increase ---")
    candidates = engineer.infer_from_beta_increase(beta_increase_pct=40.0, tolerance=10.0)
    print(f"Candidate drugs that could cause 40% beta increase:")
    for c in candidates[:5]:
        print(f"  - {c['drug']}: ~{c['estimated_concentration_uM']:.2f} uM")

    # Scenario 3: Pattern matching
    print("\n--- Scenario: Patient shows high anxiolysis, moderate sedation ---")
    pattern = {"anxiolysis": 0.85, "sedation": 0.4, "amnesia": 0.5}
    candidates = engineer.infer_from_effect_pattern(pattern)
    print(f"Drugs matching the effect pattern:")
    for c in candidates[:3]:
        print(f"  - {c['drug']}: {c['match_score']:.1%} match")


def demo_drug_interactions():
    """Demo drug-drug interaction analysis."""
    print("\n" + "=" * 80)
    print("4. DRUG INTERACTIONS - Synergy vs Competition")
    print("=" * 80)

    model = UnifiedGABAaModel(mode=ModelMode.DATABASE)

    # Interaction 1: Different sites (SYNERGY)
    print("\n--- Different binding sites (SYNERGY expected) ---")
    print("Diazepam (BZ site) + Propofol (anesthetic site)")
    result = model.simulate_interaction([("diazepam", 0.2), ("propofol", 2.0)])
    print(f"  Interaction type: {result['interaction_type'].upper()}")
    print(f"  Sites: {result['sites_involved']}")
    print(f"  Combined modulation: {result['combined_modulation']:.2f}x")
    print(f"  Sedation: {result['sedation_pct']:.1f}%")

    # Reset model
    model = UnifiedGABAaModel(mode=ModelMode.DATABASE)

    # Interaction 2: Same site (COMPETITION)
    print("\n--- Same binding site (COMPETITION expected) ---")
    print("Diazepam + Alprazolam (both BZ site)")
    result = model.simulate_interaction([("diazepam", 0.2), ("alprazolam", 0.1)])
    print(f"  Interaction type: {result['interaction_type'].upper()}")
    print(f"  Sites: {result['sites_involved']}")
    print(f"  Combined modulation: {result['combined_modulation']:.2f}x")
    print(f"  Sedation: {result['sedation_pct']:.1f}%")


def main():
    print("\n" + "#" * 80)
    print("#" + " " * 26 + "DUAL-MODE RECEPTOR SYSTEM" + " " * 27 + "#")
    print("#" * 80)

    demo_database_mode()
    demo_mechanistic_mode()
    demo_reverse_engineering()
    demo_drug_interactions()

    print("\n" + "=" * 80)
    print("SUMMARY - SYSTEM CAPABILITIES")
    print("=" * 80)
    print("""
1. DATABASE MODE
   - Uses validated drug profiles (diazepam, propofol, etc.)
   - Accurate for known drugs
   - Includes effect profiles (sedation, anxiolysis, amnesia, etc.)

2. MECHANISTIC MODE
   - First-principles prediction for novel compounds
   - Requires: Ki, efficacy, binding site
   - Predicts effects without prior data

3. REVERSE ENGINEERING
   - Given observed effect -> infer candidate drugs
   - Useful for unknown intoxication, forensics
   - Pattern matching for multi-effect scenarios

4. DRUG INTERACTIONS
   - Different sites: Multiplicative synergy
   - Same site: Competition (averaged effect)
   - Critical for polypharmacy analysis
    """)

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
