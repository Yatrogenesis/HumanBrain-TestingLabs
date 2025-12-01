#!/usr/bin/env python3
"""
Population Pharmacokinetics with CYP450 Polymorphism Variability

Implements inter-individual variability (IIV) based on:
- CYP450 metabolizer phenotypes (PM, IM, EM, UM)
- Age adjustments
- Weight-based scaling
- Hepatic/renal function

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267

References:
- Zanger UM, Schwab M (2013) Cytochrome P450 enzymes in drug metabolism.
  Pharmacol Ther 138(1):103-41
- FDA (2022) Clinical Drug Interaction Studies - Cytochrome P450
  Enzyme- and Transporter-Mediated Drug Interactions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

from pharmacokinetics import PKParameters, CompartmentalPKModel, RouteOfAdministration


class CYP450Phenotype(Enum):
    """CYP450 metabolizer phenotypes"""
    PM = "poor_metabolizer"      # 0-10% activity
    IM = "intermediate_metabolizer"  # 10-50% activity
    EM = "extensive_metabolizer"  # Normal (reference)
    UM = "ultra_metabolizer"     # >150% activity


@dataclass
class CYP450Profile:
    """CYP450 enzyme activity profile for a patient"""
    CYP2D6: CYP450Phenotype = CYP450Phenotype.EM
    CYP3A4: CYP450Phenotype = CYP450Phenotype.EM
    CYP2C19: CYP450Phenotype = CYP450Phenotype.EM
    CYP2C9: CYP450Phenotype = CYP450Phenotype.EM
    CYP1A2: CYP450Phenotype = CYP450Phenotype.EM


# CYP450 allele frequencies by population
# Reference: PharmGKB, CPIC Guidelines
CYP_POPULATION_FREQUENCIES = {
    'CYP2D6': {
        'caucasian': {CYP450Phenotype.PM: 0.07, CYP450Phenotype.IM: 0.10,
                      CYP450Phenotype.EM: 0.75, CYP450Phenotype.UM: 0.08},
        'african': {CYP450Phenotype.PM: 0.02, CYP450Phenotype.IM: 0.15,
                    CYP450Phenotype.EM: 0.60, CYP450Phenotype.UM: 0.23},
        'asian': {CYP450Phenotype.PM: 0.01, CYP450Phenotype.IM: 0.35,
                  CYP450Phenotype.EM: 0.62, CYP450Phenotype.UM: 0.02},
        'hispanic': {CYP450Phenotype.PM: 0.05, CYP450Phenotype.IM: 0.12,
                     CYP450Phenotype.EM: 0.78, CYP450Phenotype.UM: 0.05},
    },
    'CYP3A4': {
        # CYP3A4 has less genetic variation, mostly environmental/drug interactions
        'caucasian': {CYP450Phenotype.PM: 0.01, CYP450Phenotype.IM: 0.05,
                      CYP450Phenotype.EM: 0.90, CYP450Phenotype.UM: 0.04},
        'african': {CYP450Phenotype.PM: 0.01, CYP450Phenotype.IM: 0.08,
                    CYP450Phenotype.EM: 0.88, CYP450Phenotype.UM: 0.03},
        'asian': {CYP450Phenotype.PM: 0.01, CYP450Phenotype.IM: 0.06,
                  CYP450Phenotype.EM: 0.90, CYP450Phenotype.UM: 0.03},
        'hispanic': {CYP450Phenotype.PM: 0.01, CYP450Phenotype.IM: 0.05,
                     CYP450Phenotype.EM: 0.90, CYP450Phenotype.UM: 0.04},
    },
    'CYP2C19': {
        'caucasian': {CYP450Phenotype.PM: 0.03, CYP450Phenotype.IM: 0.18,
                      CYP450Phenotype.EM: 0.55, CYP450Phenotype.UM: 0.24},
        'african': {CYP450Phenotype.PM: 0.04, CYP450Phenotype.IM: 0.25,
                    CYP450Phenotype.EM: 0.55, CYP450Phenotype.UM: 0.16},
        'asian': {CYP450Phenotype.PM: 0.15, CYP450Phenotype.IM: 0.35,
                  CYP450Phenotype.EM: 0.47, CYP450Phenotype.UM: 0.03},
        'hispanic': {CYP450Phenotype.PM: 0.04, CYP450Phenotype.IM: 0.20,
                     CYP450Phenotype.EM: 0.55, CYP450Phenotype.UM: 0.21},
    },
}

# Metabolizer activity multipliers
PHENOTYPE_ACTIVITY = {
    CYP450Phenotype.PM: 0.05,   # 5% of normal activity
    CYP450Phenotype.IM: 0.35,   # 35% of normal
    CYP450Phenotype.EM: 1.0,    # Reference (100%)
    CYP450Phenotype.UM: 2.0,    # 200% of normal
}

# Drug-CYP450 metabolism mapping (fraction metabolized by each enzyme)
# Reference: FDA Drug Interaction Table
DRUG_CYP_METABOLISM = {
    'diazepam': {'CYP3A4': 0.50, 'CYP2C19': 0.40, 'CYP2C9': 0.10},
    'alprazolam': {'CYP3A4': 0.95, 'CYP2C19': 0.05},
    'midazolam': {'CYP3A4': 0.95, 'CYP2C19': 0.05},
    'triazolam': {'CYP3A4': 0.95},
    'clonazepam': {'CYP3A4': 0.60, 'CYP2C19': 0.30, 'other': 0.10},
    'lorazepam': {'glucuronidation': 0.95},  # Not CYP-dependent
    'zolpidem': {'CYP3A4': 0.60, 'CYP2C9': 0.22, 'CYP1A2': 0.14},
    'propofol': {'CYP2B6': 0.30, 'glucuronidation': 0.60, 'CYP2C9': 0.10},
    'ketamine': {'CYP3A4': 0.65, 'CYP2B6': 0.25, 'CYP2C9': 0.10},
    'fluoxetine': {'CYP2D6': 0.60, 'CYP2C9': 0.20, 'CYP3A4': 0.20},
    'morphine': {'glucuronidation': 0.90, 'CYP3A4': 0.05, 'CYP2D6': 0.05},
    'fentanyl': {'CYP3A4': 0.95},
    'haloperidol': {'CYP2D6': 0.40, 'CYP3A4': 0.50, 'glucuronidation': 0.10},
}


@dataclass
class PatientParameters:
    """Patient-specific parameters for population PK"""
    age: int = 40
    weight_kg: float = 70.0
    sex: str = 'M'  # 'M' or 'F'
    population: str = 'caucasian'
    cyp_profile: CYP450Profile = field(default_factory=CYP450Profile)

    # Organ function (0-1, where 1 is normal)
    hepatic_function: float = 1.0  # Child-Pugh score-derived
    renal_function: float = 1.0    # eGFR-derived

    # Concomitant medications (CYP inhibitors/inducers)
    cyp_inhibitors: List[str] = field(default_factory=list)
    cyp_inducers: List[str] = field(default_factory=list)


class PopulationPKModel:
    """
    Population Pharmacokinetic Model with Inter-Individual Variability (IIV).

    Implements:
    - CYP450 polymorphism effects on clearance
    - Age-related changes
    - Weight-based dosing
    - Drug-drug interactions via CYP inhibition/induction
    """

    def __init__(self, base_pk: PKParameters, patient: PatientParameters, drug_name: str):
        self.base_pk = base_pk
        self.patient = patient
        self.drug_name = drug_name

        # Adjust PK parameters for patient
        self.adjusted_pk = self._adjust_pk_for_patient()

        # Create underlying compartmental model
        self.pk_model = CompartmentalPKModel(self.adjusted_pk, patient.weight_kg)

    def _get_cyp_clearance_factor(self) -> float:
        """Calculate clearance factor based on CYP450 phenotype."""
        if self.drug_name not in DRUG_CYP_METABOLISM:
            return 1.0  # No data, assume normal

        metabolism = DRUG_CYP_METABOLISM[self.drug_name]
        total_factor = 0.0
        accounted_fraction = 0.0

        for enzyme, fraction in metabolism.items():
            if enzyme.startswith('CYP'):
                # Get phenotype for this enzyme
                phenotype = getattr(self.patient.cyp_profile, enzyme, CYP450Phenotype.EM)
                activity = PHENOTYPE_ACTIVITY[phenotype]

                # Weighted contribution to clearance
                total_factor += fraction * activity
                accounted_fraction += fraction
            else:
                # Non-CYP metabolism (glucuronidation, etc.) - assume normal
                total_factor += fraction
                accounted_fraction += fraction

        # Normalize to fraction accounted for
        if accounted_fraction > 0:
            return total_factor / accounted_fraction
        return 1.0

    def _get_age_factor(self) -> float:
        """Calculate clearance factor based on age."""
        age = self.patient.age

        if age < 18:
            # Pediatric: faster metabolism
            return 1.2
        elif age <= 65:
            # Adult: reference
            return 1.0
        elif age <= 75:
            # Elderly: reduced clearance
            return 0.85
        else:
            # Very elderly: significantly reduced
            return 0.70

    def _get_hepatic_factor(self) -> float:
        """Clearance factor based on hepatic function."""
        hf = self.patient.hepatic_function

        if hf >= 0.9:
            return 1.0
        elif hf >= 0.7:
            return 0.85  # Mild impairment
        elif hf >= 0.5:
            return 0.60  # Moderate impairment
        else:
            return 0.35  # Severe impairment

    def _get_ddi_factor(self) -> float:
        """Drug-drug interaction factor from CYP inhibitors/inducers."""
        factor = 1.0

        # Strong inhibitors (reduce clearance by 50-80%)
        strong_inhibitors = ['ketoconazole', 'itraconazole', 'ritonavir', 'clarithromycin']
        # Moderate inhibitors (reduce by 20-50%)
        moderate_inhibitors = ['fluconazole', 'erythromycin', 'diltiazem', 'verapamil']
        # Strong inducers (increase clearance by 50-80%)
        strong_inducers = ['rifampin', 'phenytoin', 'carbamazepine', 'phenobarbital']

        for inhibitor in self.patient.cyp_inhibitors:
            if inhibitor.lower() in strong_inhibitors:
                factor *= 0.30  # 70% inhibition
            elif inhibitor.lower() in moderate_inhibitors:
                factor *= 0.60  # 40% inhibition

        for inducer in self.patient.cyp_inducers:
            if inducer.lower() in strong_inducers:
                factor *= 2.5  # 150% increase in clearance

        return factor

    def _adjust_pk_for_patient(self) -> PKParameters:
        """Create patient-specific PK parameters."""
        # Calculate all adjustment factors
        cyp_factor = self._get_cyp_clearance_factor()
        age_factor = self._get_age_factor()
        hepatic_factor = self._get_hepatic_factor()
        ddi_factor = self._get_ddi_factor()

        # Combined clearance factor
        clearance_factor = cyp_factor * age_factor * hepatic_factor * ddi_factor

        # Adjust half-life inversely with clearance
        half_life_factor = 1.0 / clearance_factor if clearance_factor > 0 else 1.0

        return PKParameters(
            bioavailability=self.base_pk.bioavailability,
            absorption_rate_constant=self.base_pk.absorption_rate_constant,
            volume_of_distribution_L_kg=self.base_pk.volume_of_distribution_L_kg,
            protein_binding_fraction=self.base_pk.protein_binding_fraction,
            clearance_L_h_kg=self.base_pk.clearance_L_h_kg * clearance_factor,
            half_life_hours=self.base_pk.half_life_hours * half_life_factor,
            bbb_permeability=self.base_pk.bbb_permeability,
            brain_partition_coefficient=self.base_pk.brain_partition_coefficient,
        )

    def get_adjustment_report(self) -> Dict:
        """Get a report of all PK adjustments made."""
        return {
            'patient': {
                'age': self.patient.age,
                'weight_kg': self.patient.weight_kg,
                'sex': self.patient.sex,
                'population': self.patient.population,
                'hepatic_function': self.patient.hepatic_function,
                'renal_function': self.patient.renal_function,
            },
            'cyp_profile': {
                'CYP2D6': self.patient.cyp_profile.CYP2D6.value,
                'CYP3A4': self.patient.cyp_profile.CYP3A4.value,
                'CYP2C19': self.patient.cyp_profile.CYP2C19.value,
            },
            'adjustment_factors': {
                'cyp_factor': self._get_cyp_clearance_factor(),
                'age_factor': self._get_age_factor(),
                'hepatic_factor': self._get_hepatic_factor(),
                'ddi_factor': self._get_ddi_factor(),
            },
            'pk_changes': {
                'base_clearance': self.base_pk.clearance_L_h_kg,
                'adjusted_clearance': self.adjusted_pk.clearance_L_h_kg,
                'base_half_life': self.base_pk.half_life_hours,
                'adjusted_half_life': self.adjusted_pk.half_life_hours,
            },
            'drug_cyp_metabolism': DRUG_CYP_METABOLISM.get(self.drug_name, {}),
        }

    def simulate(self, dose_mg: float, route: RouteOfAdministration,
                 duration_hours: float = 24.0, dt: float = 0.1,
                 molecular_weight: float = 300.0) -> Dict:
        """
        Simulate PK profile with patient-specific adjustments.

        Returns:
            Dictionary with time course and adjustment report
        """
        # Administer dose
        self.pk_model.administer_dose(dose_mg, route)

        # Simulate
        n_steps = int(duration_hours / dt)
        time = np.zeros(n_steps)
        plasma_conc = np.zeros(n_steps)
        brain_conc = np.zeros(n_steps)

        for i in range(n_steps):
            concentrations = self.pk_model.step(dt)
            time[i] = concentrations["time_hours"]
            plasma_conc[i] = concentrations["plasma_mg_L"]
            brain_conc[i] = self.pk_model.get_brain_concentration_uM(molecular_weight)

        # Calculate PK metrics
        cmax_idx = np.argmax(brain_conc)
        tmax = time[cmax_idx]
        cmax = brain_conc[cmax_idx]

        # AUC (trapezoidal rule)
        auc = np.trapz(brain_conc, time)

        return {
            'time_hours': time,
            'plasma_concentration_mg_L': plasma_conc,
            'brain_concentration_uM': brain_conc,
            'metrics': {
                'Cmax_uM': cmax,
                'Tmax_hours': tmax,
                'AUC_uM_h': auc,
            },
            'adjustment_report': self.get_adjustment_report(),
        }


def generate_virtual_population(
    n_patients: int = 100,
    population: str = 'caucasian',
    age_range: Tuple[int, int] = (18, 80),
    weight_range: Tuple[float, float] = (50, 120),
    seed: int = 42
) -> List[PatientParameters]:
    """
    Generate a virtual population for population PK analysis.

    Args:
        n_patients: Number of virtual patients
        population: Ethnic background for CYP frequencies
        age_range: (min_age, max_age)
        weight_range: (min_weight_kg, max_weight_kg)
        seed: Random seed for reproducibility

    Returns:
        List of PatientParameters
    """
    np.random.seed(seed)
    patients = []

    for _ in range(n_patients):
        # Generate demographics
        age = np.random.randint(age_range[0], age_range[1] + 1)
        weight = np.random.uniform(weight_range[0], weight_range[1])
        sex = np.random.choice(['M', 'F'])

        # Generate CYP450 profile based on population frequencies
        cyp_profile = CYP450Profile()

        for enzyme in ['CYP2D6', 'CYP3A4', 'CYP2C19']:
            if enzyme in CYP_POPULATION_FREQUENCIES:
                freqs = CYP_POPULATION_FREQUENCIES[enzyme].get(population, {})
                if freqs:
                    phenotypes = list(freqs.keys())
                    probs = list(freqs.values())
                    chosen = np.random.choice(phenotypes, p=probs)
                    setattr(cyp_profile, enzyme, chosen)

        # Hepatic function (mostly normal with some impairment)
        hepatic = np.random.choice(
            [1.0, 0.85, 0.65, 0.45],
            p=[0.85, 0.10, 0.04, 0.01]
        )

        patients.append(PatientParameters(
            age=age,
            weight_kg=weight,
            sex=sex,
            population=population,
            cyp_profile=cyp_profile,
            hepatic_function=hepatic,
        ))

    return patients


def run_population_simulation(
    drug_name: str,
    base_pk: PKParameters,
    dose_mg: float,
    route: RouteOfAdministration,
    molecular_weight: float,
    n_patients: int = 100,
    duration_hours: float = 24.0
) -> Dict:
    """
    Run population PK simulation.

    Returns:
        Dictionary with population statistics and individual data
    """
    # Generate population
    population = generate_virtual_population(n_patients=n_patients)

    # Run simulations
    all_cmax = []
    all_tmax = []
    all_auc = []
    individual_results = []

    for patient in population:
        model = PopulationPKModel(base_pk, patient, drug_name)
        result = model.simulate(
            dose_mg=dose_mg,
            route=route,
            duration_hours=duration_hours,
            molecular_weight=molecular_weight
        )

        all_cmax.append(result['metrics']['Cmax_uM'])
        all_tmax.append(result['metrics']['Tmax_hours'])
        all_auc.append(result['metrics']['AUC_uM_h'])

        individual_results.append({
            'patient': model.get_adjustment_report()['patient'],
            'metrics': result['metrics'],
        })

    # Population statistics
    return {
        'drug': drug_name,
        'dose_mg': dose_mg,
        'route': route.value,
        'n_patients': n_patients,
        'population_statistics': {
            'Cmax_uM': {
                'mean': np.mean(all_cmax),
                'std': np.std(all_cmax),
                'min': np.min(all_cmax),
                'max': np.max(all_cmax),
                'cv_percent': (np.std(all_cmax) / np.mean(all_cmax)) * 100,
            },
            'Tmax_hours': {
                'mean': np.mean(all_tmax),
                'std': np.std(all_tmax),
            },
            'AUC_uM_h': {
                'mean': np.mean(all_auc),
                'std': np.std(all_auc),
                'cv_percent': (np.std(all_auc) / np.mean(all_auc)) * 100,
            },
        },
        'percentiles': {
            'Cmax_5th': np.percentile(all_cmax, 5),
            'Cmax_95th': np.percentile(all_cmax, 95),
            'AUC_5th': np.percentile(all_auc, 5),
            'AUC_95th': np.percentile(all_auc, 95),
        },
    }


if __name__ == '__main__':
    print("=" * 80)
    print("POPULATION PHARMACOKINETICS - CYP450 VARIABILITY TEST")
    print("=" * 80)

    from pharmacokinetics import DrugPKDatabase

    # Test with diazepam (CYP3A4 + CYP2C19 substrate)
    db = DrugPKDatabase()
    base_pk = db.get_diazepam_pk()

    print("\n--- DIAZEPAM 10mg ORAL ---")
    print("\n1. Normal Metabolizer (EM)")
    patient_em = PatientParameters(
        age=40, weight_kg=70,
        cyp_profile=CYP450Profile(CYP3A4=CYP450Phenotype.EM, CYP2C19=CYP450Phenotype.EM)
    )
    model_em = PopulationPKModel(base_pk, patient_em, 'diazepam')
    result_em = model_em.simulate(10, RouteOfAdministration.ORAL, duration_hours=72, molecular_weight=284.74)
    print(f"   Cmax: {result_em['metrics']['Cmax_uM']:.3f} uM")
    print(f"   Tmax: {result_em['metrics']['Tmax_hours']:.2f} h")

    print("\n2. Poor Metabolizer (PM) - CYP2C19")
    patient_pm = PatientParameters(
        age=40, weight_kg=70,
        cyp_profile=CYP450Profile(CYP3A4=CYP450Phenotype.EM, CYP2C19=CYP450Phenotype.PM)
    )
    model_pm = PopulationPKModel(base_pk, patient_pm, 'diazepam')
    result_pm = model_pm.simulate(10, RouteOfAdministration.ORAL, duration_hours=72, molecular_weight=284.74)
    print(f"   Cmax: {result_pm['metrics']['Cmax_uM']:.3f} uM ({result_pm['metrics']['Cmax_uM']/result_em['metrics']['Cmax_uM']*100:.0f}% of EM)")
    print(f"   Tmax: {result_pm['metrics']['Tmax_hours']:.2f} h")

    print("\n3. Ultra Metabolizer (UM) - CYP3A4")
    patient_um = PatientParameters(
        age=40, weight_kg=70,
        cyp_profile=CYP450Profile(CYP3A4=CYP450Phenotype.UM, CYP2C19=CYP450Phenotype.EM)
    )
    model_um = PopulationPKModel(base_pk, patient_um, 'diazepam')
    result_um = model_um.simulate(10, RouteOfAdministration.ORAL, duration_hours=72, molecular_weight=284.74)
    print(f"   Cmax: {result_um['metrics']['Cmax_uM']:.3f} uM ({result_um['metrics']['Cmax_uM']/result_em['metrics']['Cmax_uM']*100:.0f}% of EM)")
    print(f"   Tmax: {result_um['metrics']['Tmax_hours']:.2f} h")

    print("\n4. Elderly (80 years) + Hepatic Impairment")
    patient_elderly = PatientParameters(
        age=80, weight_kg=60,
        hepatic_function=0.65
    )
    model_elderly = PopulationPKModel(base_pk, patient_elderly, 'diazepam')
    result_elderly = model_elderly.simulate(10, RouteOfAdministration.ORAL, duration_hours=72, molecular_weight=284.74)
    print(f"   Cmax: {result_elderly['metrics']['Cmax_uM']:.3f} uM ({result_elderly['metrics']['Cmax_uM']/result_em['metrics']['Cmax_uM']*100:.0f}% of EM)")
    print(f"   Tmax: {result_elderly['metrics']['Tmax_hours']:.2f} h")

    print("\n--- POPULATION SIMULATION (100 patients) ---")
    pop_result = run_population_simulation(
        'diazepam', base_pk, 10, RouteOfAdministration.ORAL, 284.74, n_patients=100
    )
    stats = pop_result['population_statistics']
    print(f"   Cmax: {stats['Cmax_uM']['mean']:.3f} ± {stats['Cmax_uM']['std']:.3f} uM (CV: {stats['Cmax_uM']['cv_percent']:.1f}%)")
    print(f"   AUC:  {stats['AUC_uM_h']['mean']:.1f} ± {stats['AUC_uM_h']['std']:.1f} uM*h (CV: {stats['AUC_uM_h']['cv_percent']:.1f}%)")
    print(f"   5th-95th percentile Cmax: {pop_result['percentiles']['Cmax_5th']:.3f} - {pop_result['percentiles']['Cmax_95th']:.3f} uM")

    print("\n" + "=" * 80)
    print("Population PK module test complete!")
