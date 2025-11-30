#!/usr/bin/env python3
"""
Complete Drug Simulation
========================

Simulates all drugs in the DRUG_PROFILES database and generates
comprehensive results including:
- Dose-response curves
- Effect profiles
- Drug comparisons by class
- Interaction predictions

Author: Francisco Molina Burgos (Yatrogenesis)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pharmacology.receptor_mechanisms import (
    UnifiedGABAaModel, ModelMode, BindingSite, EffectType,
    DRUG_PROFILES, DrugMolecularProfile
)


def simulate_single_drug(drug_name: str, concentrations: List[float] = None) -> Dict:
    """Simulate a single drug across multiple concentrations."""
    if concentrations is None:
        # Default concentration range based on drug type
        profile = DRUG_PROFILES[drug_name]
        ki_uM = profile.ki_nM / 1000.0
        # Range from 0.1x Ki to 100x Ki
        concentrations = [ki_uM * mult for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]]

    model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
    results = []

    for conc in concentrations:
        result = model.simulate(drug_name=drug_name, concentration_uM=conc)
        results.append({
            "concentration_uM": conc,
            "occupancy": result["occupancy"],
            "modulation": result["modulation"],
            "beta_increase_pct": result["beta_increase_pct"],
            "sedation_pct": result["sedation_pct"],
            "effects": result["effects"],
        })

    return {
        "drug": drug_name,
        "profile": {
            "binding_site": DRUG_PROFILES[drug_name].binding_site.value,
            "ki_nM": DRUG_PROFILES[drug_name].ki_nM,
            "efficacy": DRUG_PROFILES[drug_name].intrinsic_efficacy,
            "molecular_weight": DRUG_PROFILES[drug_name].molecular_weight,
            "logP": DRUG_PROFILES[drug_name].logP,
        },
        "dose_response": results,
    }


def simulate_all_drugs() -> Dict:
    """Simulate all drugs in the database."""
    print("\n" + "=" * 80)
    print("COMPLETE DRUG SIMULATION")
    print(f"Database contains {len(DRUG_PROFILES)} drugs")
    print("=" * 80)

    all_results = {}
    drug_classes = {}

    # Group drugs by binding site
    for drug_name, profile in DRUG_PROFILES.items():
        site = profile.binding_site.value
        if site not in drug_classes:
            drug_classes[site] = []
        drug_classes[site].append(drug_name)

    # Simulate each class
    for site, drugs in drug_classes.items():
        print(f"\n--- {site.upper()} SITE ({len(drugs)} drugs) ---")

        for drug in drugs:
            result = simulate_single_drug(drug)
            all_results[drug] = result

            # Get therapeutic concentration estimate
            profile = DRUG_PROFILES[drug]
            therapeutic_conc = profile.ki_nM / 1000.0 * 5  # 5x Ki estimate

            # Simulate at therapeutic concentration
            model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
            therapeutic_result = model.simulate(drug_name=drug, concentration_uM=therapeutic_conc)

            print(f"  {drug:15} | Ki={profile.ki_nM:8.1f} nM | "
                  f"Eff={profile.intrinsic_efficacy:.2f} | "
                  f"@{therapeutic_conc:.2f}uM: Beta={therapeutic_result['beta_increase_pct']:5.1f}% "
                  f"Sed={therapeutic_result['sedation_pct']:5.1f}%")

    return {
        "timestamp": datetime.now().isoformat(),
        "total_drugs": len(DRUG_PROFILES),
        "drug_classes": {k: len(v) for k, v in drug_classes.items()},
        "results": all_results,
    }


def generate_comparison_table() -> str:
    """Generate a comparison table of all drugs."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("DRUG COMPARISON TABLE")
    lines.append("=" * 100)

    # Header
    header = f"{'Drug':<15} {'Site':<12} {'Ki(nM)':<10} {'Eff':<6} {'Sed':<6} {'Anx':<6} {'Amn':<6} {'Anesth':<6} {'Anticonv':<8}"
    lines.append(header)
    lines.append("-" * 100)

    # Sort by binding site
    sorted_drugs = sorted(DRUG_PROFILES.items(), key=lambda x: (x[1].binding_site.value, x[0]))

    for drug_name, profile in sorted_drugs:
        effects = profile.effect_profile
        sed = effects.get(EffectType.SEDATION, 0) * 100
        anx = effects.get(EffectType.ANXIOLYSIS, 0) * 100
        amn = effects.get(EffectType.AMNESIA, 0) * 100
        anes = effects.get(EffectType.ANESTHESIA, 0) * 100
        anti = effects.get(EffectType.ANTICONVULSANT, 0) * 100

        line = f"{drug_name:<15} {profile.binding_site.value:<12} {profile.ki_nM:<10.1f} {profile.intrinsic_efficacy:<6.2f} {sed:<6.0f} {anx:<6.0f} {amn:<6.0f} {anes:<6.0f} {anti:<8.0f}"
        lines.append(line)

    lines.append("=" * 100)
    return "\n".join(lines)


def simulate_clinical_scenarios():
    """Simulate clinically relevant scenarios."""
    print("\n" + "=" * 80)
    print("CLINICAL SCENARIOS")
    print("=" * 80)

    scenarios = [
        # Anxiolytic comparison
        {
            "name": "Anxiolytic Therapy (1x Ki)",
            "drugs": ["diazepam", "alprazolam", "lorazepam", "bromazepam"],
            "conc_multiplier": 1.0,
        },
        # Sedation for procedures
        {
            "name": "Procedural Sedation (5x Ki)",
            "drugs": ["midazolam", "propofol", "etomidate"],
            "conc_multiplier": 5.0,
        },
        # General anesthesia
        {
            "name": "General Anesthesia (10x Ki)",
            "drugs": ["propofol", "sevoflurane", "isoflurane", "thiopental"],
            "conc_multiplier": 10.0,
        },
        # Anticonvulsant
        {
            "name": "Anticonvulsant Therapy (2x Ki)",
            "drugs": ["clonazepam", "lorazepam", "phenobarbital"],
            "conc_multiplier": 2.0,
        },
        # Hypnotic comparison
        {
            "name": "Hypnotic for Sleep (2x Ki)",
            "drugs": ["zolpidem", "zaleplon", "triazolam", "temazepam"],
            "conc_multiplier": 2.0,
        },
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        scenario_results = {}

        for drug in scenario["drugs"]:
            if drug not in DRUG_PROFILES:
                print(f"  {drug}: Not in database")
                continue

            profile = DRUG_PROFILES[drug]
            conc = (profile.ki_nM / 1000.0) * scenario["conc_multiplier"]

            model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
            result = model.simulate(drug_name=drug, concentration_uM=conc)

            scenario_results[drug] = {
                "concentration_uM": conc,
                "occupancy": result["occupancy"],
                "modulation": result["modulation"],
                "sedation_pct": result["sedation_pct"],
                "effects": result["effects"],
            }

            print(f"  {drug:15} @ {conc:7.2f} uM: "
                  f"Occ={result['occupancy']:.0%} | "
                  f"Mod={result['modulation']:.2f}x | "
                  f"Sed={result['sedation_pct']:.0f}%")

        results[scenario["name"]] = scenario_results

    return results


def simulate_interactions():
    """Simulate drug-drug interactions."""
    print("\n" + "=" * 80)
    print("DRUG-DRUG INTERACTIONS")
    print("=" * 80)

    interactions = [
        # Synergy (different sites)
        {
            "name": "BZ + Anesthetic (Synergy)",
            "drugs": [("midazolam", 0.05), ("propofol", 2.0)],
        },
        {
            "name": "BZ + Volatile (MAC-sparing)",
            "drugs": [("diazepam", 0.1), ("sevoflurane", 300.0)],
        },
        # Competition (same site)
        {
            "name": "BZ + BZ (Competition)",
            "drugs": [("diazepam", 0.2), ("alprazolam", 0.05)],
        },
        # Triple combination
        {
            "name": "BZ + Propofol + Barb",
            "drugs": [("midazolam", 0.03), ("propofol", 1.0), ("thiopental", 5.0)],
        },
    ]

    results = {}

    for interaction in interactions:
        print(f"\n--- {interaction['name']} ---")

        model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
        result = model.simulate_interaction(interaction["drugs"])

        results[interaction["name"]] = result

        print(f"  Drugs: {interaction['drugs']}")
        print(f"  Sites: {result['sites_involved']}")
        print(f"  Type: {result['interaction_type'].upper()}")
        print(f"  Combined modulation: {result['combined_modulation']:.2f}x")
        print(f"  Sedation: {result['sedation_pct']:.1f}%")

    return results


def main():
    print("\n" + "#" * 80)
    print("#" + " " * 25 + "COMPLETE DRUG SIMULATION" + " " * 25 + "#")
    print("#" * 80)

    # 1. Simulate all drugs
    all_results = simulate_all_drugs()

    # 2. Generate comparison table
    comparison_table = generate_comparison_table()
    print(comparison_table)

    # 3. Clinical scenarios
    clinical_results = simulate_clinical_scenarios()

    # 4. Drug interactions
    interaction_results = simulate_interactions()

    # 5. Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "drug_simulation_complete.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "all_drugs": all_results,
            "clinical_scenarios": clinical_results,
            "interactions": interaction_results,
        }, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Total drugs simulated: {len(DRUG_PROFILES)}")
    print(f"Drug classes:")
    for site, count in all_results["drug_classes"].items():
        print(f"  - {site}: {count} drugs")
    print(f"Clinical scenarios: {len(clinical_results)}")
    print(f"Interactions tested: {len(interaction_results)}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
