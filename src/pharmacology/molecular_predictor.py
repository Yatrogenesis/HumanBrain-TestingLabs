#!/usr/bin/env python3
"""
Molecular Property-Based PK Parameter Predictor
================================================

Automatically derives pharmacokinetic parameters from molecular descriptors
using established QSAR/PBPK relationships.

This enables simulation of novel drugs without manual parameter calibration.

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267

References:
- Lipinski CA et al. (2001) Experimental and computational approaches
  to estimate solubility and permeability. Adv Drug Deliv Rev 46:3-26
- Waring MJ (2010) Lipophilicity in drug discovery. Expert Opin Drug Discov 5:235
- Di L, Kerns EH (2016) Drug-Like Properties: Concepts, Structure Design
  and Methods from ADME to Toxicity Optimization. Academic Press.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
import math


class TransporterType(Enum):
    """BBB transporter types that can affect brain penetration."""
    PASSIVE = "passive"  # Lipophilic passive diffusion
    LAT1 = "lat1"  # Large neutral amino acid transporter (L-DOPA)
    OATP = "oatp"  # Organic anion transporting polypeptide
    PGLP = "p-gp"  # P-glycoprotein (efflux)
    BCRP = "bcrp"  # Breast cancer resistance protein (efflux)


class DrugClass(Enum):
    """Drug classes with characteristic PK behavior."""
    GENERAL_ANESTHETIC = "anesthetic"
    DISSOCIATIVE = "dissociative"
    ANTIDEPRESSANT_SSRI = "ssri"
    ANTIDEPRESSANT_TCA = "tca"
    ANXIOLYTIC_BZ = "benzodiazepine"
    DOPAMINE_PRECURSOR = "dopamine_precursor"
    DOPAMINE_AGONIST = "dopamine_agonist"
    ANTIPSYCHOTIC = "antipsychotic"
    OPIOID = "opioid"
    STIMULANT = "stimulant"
    ANTICONVULSANT = "anticonvulsant"
    UNKNOWN = "unknown"


@dataclass
class MolecularDescriptors:
    """
    Molecular descriptors used for PK prediction.

    These can be calculated from structure using RDKit or obtained
    from databases like PubChem, DrugBank, or ChEMBL.
    """
    # Basic properties
    molecular_weight: float  # g/mol (Da)
    logP: float  # Octanol-water partition coefficient

    # Polar descriptors
    psa: float = 60.0  # Polar surface area (Å²) - default for CNS drugs
    hbd: int = 2  # H-bond donors
    hba: int = 4  # H-bond acceptors

    # Ionization
    pKa_acid: Optional[float] = None  # Acidic pKa
    pKa_base: Optional[float] = None  # Basic pKa

    # Structure
    rotatable_bonds: int = 5
    aromatic_rings: int = 2

    # Drug class (helps with receptor targeting)
    drug_class: DrugClass = DrugClass.UNKNOWN

    # Transporter information
    primary_transporter: TransporterType = TransporterType.PASSIVE
    pgp_substrate: bool = False  # P-gp efflux reduces brain penetration

    # Optional: Known receptor targets with Ki values
    receptor_targets: Dict[str, float] = field(default_factory=dict)
    # Format: {"GABA_A": 3.5, "5HT2A": 15.0} (Ki in nM)


class MolecularPKPredictor:
    """
    Predicts pharmacokinetic parameters from molecular descriptors.

    Uses established QSAR relationships and PBPK scaling factors.

    Example usage:
    ```python
    descriptors = MolecularDescriptors(
        molecular_weight=178.27,
        logP=3.8,
        psa=17.07,
        hbd=1,
        hba=1,
        drug_class=DrugClass.GENERAL_ANESTHETIC,
    )

    predictor = MolecularPKPredictor()
    pk_params = predictor.predict(descriptors)
    ```
    """

    def __init__(self, body_weight_kg: float = 70.0):
        """
        Initialize predictor.

        Args:
            body_weight_kg: Reference body weight for scaling
        """
        self.body_weight = body_weight_kg

        # Lipinski's Rule of 5 thresholds (CNS drugs are stricter)
        self.ro5_cns = {
            "mw_max": 450,  # Lower than standard 500
            "logp_max": 5,
            "logp_min": 1,  # CNS needs some lipophilicity
            "psa_max": 90,  # Lower than standard 140
            "hbd_max": 3,
            "hba_max": 7,
        }

    def predict(self, mol: MolecularDescriptors) -> Dict[str, Any]:
        """
        Predict all PK parameters from molecular descriptors.

        Args:
            mol: MolecularDescriptors object

        Returns:
            Dictionary with predicted PKParameters values
        """
        # Predict individual parameters
        bioavailability = self._predict_bioavailability(mol)
        absorption_rate = self._predict_absorption_rate(mol)
        vd = self._predict_volume_of_distribution(mol)
        protein_binding = self._predict_protein_binding(mol)
        clearance = self._predict_clearance(mol)
        half_life = self._predict_half_life(vd, clearance)
        bbb_permeability = self._predict_bbb_permeability(mol)
        brain_partition = self._predict_brain_partition(mol)

        return {
            "bioavailability": bioavailability,
            "absorption_rate_constant": absorption_rate,
            "volume_of_distribution_L_kg": vd,
            "protein_binding_fraction": protein_binding,
            "clearance_L_h_kg": clearance,
            "half_life_hours": half_life,
            "bbb_permeability": bbb_permeability,
            "brain_partition_coefficient": brain_partition,
            # Additional derived parameters
            "cns_mpo_score": self._calculate_cns_mpo(mol),
            "lipinski_violations": self._count_lipinski_violations(mol),
        }

    def _predict_bioavailability(self, mol: MolecularDescriptors) -> float:
        """
        Predict oral bioavailability (F).

        Based on Veber rules and PSA correlation.
        Reference: Veber DF et al. (2002) J Med Chem 45:2615
        """
        # Base bioavailability from PSA
        # PSA < 75: ~90% F
        # PSA 75-140: linear decrease
        # PSA > 140: ~10% F
        if mol.psa < 75:
            f_psa = 0.90
        elif mol.psa < 140:
            f_psa = 0.90 - 0.012 * (mol.psa - 75)
        else:
            f_psa = 0.10

        # Rotatable bonds penalty
        # >10 rotatable bonds reduces F
        f_rot = 1.0 if mol.rotatable_bonds <= 10 else 0.8 ** (mol.rotatable_bonds - 10)

        # LogP correction (too high or too low reduces F)
        if 0 < mol.logP < 5:
            f_logp = 1.0
        elif mol.logP >= 5:
            f_logp = 0.5  # Solubility limited
        else:
            f_logp = 0.6  # Permeability limited

        # Drug class adjustments
        class_factor = {
            DrugClass.DOPAMINE_PRECURSOR: 0.30,  # First-pass metabolism
            DrugClass.OPIOID: 0.40,  # First-pass metabolism
            DrugClass.GENERAL_ANESTHETIC: 0.95,  # Usually IV but some oral
            DrugClass.ANTIDEPRESSANT_SSRI: 0.70,  # Moderate first-pass
            DrugClass.ANXIOLYTIC_BZ: 0.85,  # Good absorption
        }.get(mol.drug_class, 1.0)

        return min(0.98, f_psa * f_rot * f_logp * class_factor)

    def _predict_absorption_rate(self, mol: MolecularDescriptors) -> float:
        """
        Predict oral absorption rate constant (ka, 1/h).

        Based on molecular size and lipophilicity.
        """
        # Base rate from MW (smaller = faster)
        if mol.molecular_weight < 200:
            ka_base = 3.0
        elif mol.molecular_weight < 400:
            ka_base = 2.0
        else:
            ka_base = 1.0

        # LogP adjustment (optimal ~2 for passive diffusion)
        if 1 < mol.logP < 3:
            ka_logp = 1.0
        else:
            ka_logp = 0.7

        return ka_base * ka_logp

    def _predict_volume_of_distribution(self, mol: MolecularDescriptors) -> float:
        """
        Predict volume of distribution (Vd, L/kg).

        Based on lipophilicity and ionization.
        Reference: Obach RS et al. (2008) Drug Metab Dispos 36:1385
        """
        # Base Vd from logP (more lipophilic = larger Vd)
        # Vd = 0.05 * 10^(0.5 * logP) approximately
        vd_base = 0.05 * (10 ** (0.4 * mol.logP))

        # Cap at reasonable values
        vd = max(0.1, min(20.0, vd_base))

        # Basic drugs distribute more (tissue binding)
        if mol.pKa_base is not None and mol.pKa_base > 7:
            vd *= 1.5

        return vd

    def _predict_protein_binding(self, mol: MolecularDescriptors) -> float:
        """
        Predict plasma protein binding fraction.

        Based on lipophilicity and charge.
        Reference: Yamazaki K, Kanaoka M (2004) J Pharm Sci 93:1480
        """
        # LogP-based correlation for albumin binding
        # Higher logP = more binding
        fu_logp = 1 / (1 + 10 ** (0.5 * mol.logP - 1))  # Unbound fraction

        # Acidic compounds bind more strongly to albumin
        if mol.pKa_acid is not None and mol.pKa_acid < 6:
            fu_logp *= 0.5  # Increase binding

        # Convert to bound fraction
        fb = 1 - fu_logp

        # Drug class adjustments
        class_adjustments = {
            DrugClass.GENERAL_ANESTHETIC: 0.97,  # Propofol: very high binding
            DrugClass.ANXIOLYTIC_BZ: 0.95,  # BZs: high binding
            DrugClass.ANTIDEPRESSANT_SSRI: 0.94,  # SSRIs: high binding
        }

        if mol.drug_class in class_adjustments:
            fb = class_adjustments[mol.drug_class]

        return min(0.99, max(0.01, fb))

    def _predict_clearance(self, mol: MolecularDescriptors) -> float:
        """
        Predict hepatic clearance (CL, L/h/kg).

        Based on metabolic susceptibility markers.
        """
        # Base clearance from MW and aromatic rings
        if mol.molecular_weight < 300:
            cl_base = 0.8
        else:
            cl_base = 0.5

        # Aromatic rings increase CYP metabolism
        cl_aromatic = cl_base * (1 + 0.1 * mol.aromatic_rings)

        # LogP correction (moderate lipophilicity optimal for metabolism)
        if 2 < mol.logP < 4:
            cl_logp = 1.2
        else:
            cl_logp = 1.0

        return cl_base * cl_logp

    def _predict_half_life(self, vd: float, clearance: float) -> float:
        """
        Calculate half-life from Vd and clearance.

        t1/2 = 0.693 * Vd / CL
        """
        t_half = 0.693 * vd * self.body_weight / (clearance * self.body_weight)
        return max(0.5, min(100, t_half))

    def _predict_bbb_permeability(self, mol: MolecularDescriptors) -> float:
        """
        Predict blood-brain barrier permeability (0-1 scale).

        Based on CNS MPO score and transporter effects.
        Reference: Wager TT et al. (2010) ACS Chem Neurosci 1:435
        """
        # PSA is the strongest predictor
        # PSA < 60: excellent penetration
        # PSA 60-90: good penetration
        # PSA > 90: poor penetration
        if mol.psa < 60:
            p_psa = 0.95
        elif mol.psa < 90:
            p_psa = 0.95 - 0.015 * (mol.psa - 60)
        else:
            p_psa = 0.30

        # LogP adjustment (optimal 2-4 for BBB)
        if 2 < mol.logP < 4:
            p_logp = 1.0
        elif mol.logP < 2:
            p_logp = 0.6  # Too hydrophilic
        else:
            p_logp = 0.7  # Too lipophilic (P-gp substrate risk)

        # Molecular weight penalty
        if mol.molecular_weight < 400:
            p_mw = 1.0
        elif mol.molecular_weight < 500:
            p_mw = 0.8
        else:
            p_mw = 0.5

        # H-bond donors penalty (critical for BBB)
        p_hbd = 1.0 if mol.hbd <= 2 else 0.7 ** (mol.hbd - 2)

        # Transporter effects
        transporter_factor = {
            TransporterType.PASSIVE: 1.0,
            TransporterType.LAT1: 1.2,  # Active uptake enhances
            TransporterType.OATP: 1.1,
            TransporterType.PGLP: 0.3,  # Efflux reduces
            TransporterType.BCRP: 0.4,
        }[mol.primary_transporter]

        # P-gp substrate reduces penetration
        if mol.pgp_substrate:
            transporter_factor *= 0.5

        base_permeability = p_psa * p_logp * p_mw * p_hbd * transporter_factor
        return min(0.99, max(0.01, base_permeability))

    def _predict_brain_partition(self, mol: MolecularDescriptors) -> float:
        """
        Predict brain-to-plasma partition coefficient (Kp,brain).

        Based on lipophilicity, ionization, and efflux.
        """
        # Base Kp from logP
        # Higher logP = more brain accumulation (up to a point)
        if mol.logP < 1:
            kp_base = 0.5
        elif mol.logP < 4:
            kp_base = 1.0 + 0.5 * mol.logP
        else:
            kp_base = 2.5  # Plateau (efflux kicks in)

        # Transporter effects on accumulation
        if mol.primary_transporter == TransporterType.LAT1:
            kp_base *= 10  # Active transport dramatically increases brain levels
        elif mol.pgp_substrate:
            kp_base *= 0.2  # Efflux reduces accumulation

        # Drug class adjustments (based on known brain penetration)
        class_kp = {
            DrugClass.GENERAL_ANESTHETIC: 5.0,  # High brain uptake needed
            DrugClass.DISSOCIATIVE: 3.0,
            DrugClass.ANTIDEPRESSANT_SSRI: 1.5,  # Moderate accumulation
            DrugClass.ANXIOLYTIC_BZ: 3.0,  # Good brain uptake
            DrugClass.DOPAMINE_PRECURSOR: 25.0,  # LAT1 active transport
        }

        if mol.drug_class in class_kp:
            kp_base = class_kp[mol.drug_class]

        return max(0.1, min(30.0, kp_base))

    def _calculate_cns_mpo(self, mol: MolecularDescriptors) -> float:
        """
        Calculate CNS Multi-Parameter Optimization score.

        Reference: Wager TT et al. (2016) ACS Chem Neurosci 7:767

        Score 0-6, higher = better CNS drug candidate
        """
        score = 0.0

        # MW: 250-350 optimal
        if 250 <= mol.molecular_weight <= 350:
            score += 1.0
        elif 200 <= mol.molecular_weight <= 400:
            score += 0.5

        # LogP: 2-4 optimal
        if 2 <= mol.logP <= 4:
            score += 1.0
        elif 1 <= mol.logP <= 5:
            score += 0.5

        # PSA: 40-60 optimal
        if 40 <= mol.psa <= 60:
            score += 1.0
        elif 20 <= mol.psa <= 90:
            score += 0.5

        # HBD: 0-1 optimal
        if mol.hbd <= 1:
            score += 1.0
        elif mol.hbd == 2:
            score += 0.5

        # pKa: 7.5-9.5 optimal for basic drugs
        if mol.pKa_base is not None:
            if 7.5 <= mol.pKa_base <= 9.5:
                score += 1.0
            elif 6.5 <= mol.pKa_base <= 10.5:
                score += 0.5
        else:
            score += 0.5  # Neutral is OK

        # LogD7.4 approximation (logP corrected for ionization)
        # Assume logD ≈ logP for neutrals
        if 1 <= mol.logP <= 3:
            score += 1.0
        elif 0 <= mol.logP <= 4:
            score += 0.5

        return score

    def _count_lipinski_violations(self, mol: MolecularDescriptors) -> int:
        """Count Lipinski Rule of 5 violations (CNS-specific thresholds)."""
        violations = 0

        if mol.molecular_weight > self.ro5_cns["mw_max"]:
            violations += 1
        if mol.logP > self.ro5_cns["logp_max"] or mol.logP < self.ro5_cns["logp_min"]:
            violations += 1
        if mol.psa > self.ro5_cns["psa_max"]:
            violations += 1
        if mol.hbd > self.ro5_cns["hbd_max"]:
            violations += 1
        if mol.hba > self.ro5_cns["hba_max"]:
            violations += 1

        return violations


# =============================================================================
# Pre-defined molecular descriptors for gold standard drugs
# =============================================================================

GOLD_STANDARD_MOLECULES = {
    "propofol": MolecularDescriptors(
        molecular_weight=178.27,
        logP=3.79,
        psa=20.23,
        hbd=1,
        hba=1,
        pKa_acid=11.1,  # Phenolic OH
        rotatable_bonds=2,
        aromatic_rings=1,
        drug_class=DrugClass.GENERAL_ANESTHETIC,
        primary_transporter=TransporterType.PASSIVE,
        receptor_targets={"GABA_A": 3.5},  # EC50 in μM
    ),

    "ketamine": MolecularDescriptors(
        molecular_weight=237.73,
        logP=2.75,
        psa=29.10,
        hbd=1,
        hba=2,
        pKa_base=7.5,
        rotatable_bonds=2,
        aromatic_rings=1,
        drug_class=DrugClass.DISSOCIATIVE,
        primary_transporter=TransporterType.PASSIVE,
        receptor_targets={"NMDA": 0.5},  # Ki in μM
    ),

    "levodopa": MolecularDescriptors(
        molecular_weight=197.19,
        logP=-2.74,  # Very hydrophilic amino acid
        psa=103.78,
        hbd=4,
        hba=5,
        pKa_acid=2.3,  # Carboxylic acid
        pKa_base=8.7,  # Amino group
        rotatable_bonds=3,
        aromatic_rings=1,
        drug_class=DrugClass.DOPAMINE_PRECURSOR,
        primary_transporter=TransporterType.LAT1,  # Active transport!
        pgp_substrate=False,
    ),

    "fluoxetine": MolecularDescriptors(
        molecular_weight=309.33,
        logP=4.05,
        psa=21.26,
        hbd=1,
        hba=2,
        pKa_base=10.0,
        rotatable_bonds=6,
        aromatic_rings=2,
        drug_class=DrugClass.ANTIDEPRESSANT_SSRI,
        primary_transporter=TransporterType.PASSIVE,
        pgp_substrate=True,  # Moderate P-gp substrate
        receptor_targets={"SERT": 0.9},  # Ki in nM
    ),

    "diazepam": MolecularDescriptors(
        molecular_weight=284.74,
        logP=2.82,
        psa=32.67,
        hbd=0,
        hba=3,
        pKa_base=3.4,  # Weak base
        rotatable_bonds=1,
        aromatic_rings=2,
        drug_class=DrugClass.ANXIOLYTIC_BZ,
        primary_transporter=TransporterType.PASSIVE,
        receptor_targets={"GABA_A_BZ": 0.015},  # Ki in μM (high affinity)
    ),
}


def predict_pk_from_structure(drug_name: str = None,
                               descriptors: MolecularDescriptors = None) -> Dict[str, Any]:
    """
    Convenience function to predict PK parameters.

    Args:
        drug_name: Name of gold standard drug (if available)
        descriptors: Custom MolecularDescriptors object

    Returns:
        Predicted PK parameters dictionary
    """
    predictor = MolecularPKPredictor()

    if drug_name and drug_name.lower() in GOLD_STANDARD_MOLECULES:
        mol = GOLD_STANDARD_MOLECULES[drug_name.lower()]
    elif descriptors:
        mol = descriptors
    else:
        raise ValueError("Must provide drug_name or descriptors")

    return predictor.predict(mol)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MOLECULAR PK PARAMETER PREDICTOR")
    print("=" * 80 + "\n")

    predictor = MolecularPKPredictor()

    for drug_name, mol in GOLD_STANDARD_MOLECULES.items():
        pk = predictor.predict(mol)

        print(f"\n{drug_name.upper()}")
        print("-" * 40)
        print(f"  MW: {mol.molecular_weight:.1f} Da | LogP: {mol.logP:.2f} | PSA: {mol.psa:.1f} Å²")
        print(f"  HBD: {mol.hbd} | HBA: {mol.hba} | Transport: {mol.primary_transporter.value}")
        print(f"\n  Predicted PK:")
        print(f"    Bioavailability: {pk['bioavailability']:.0%}")
        print(f"    Vd: {pk['volume_of_distribution_L_kg']:.2f} L/kg")
        print(f"    Protein binding: {pk['protein_binding_fraction']:.0%}")
        print(f"    Half-life: {pk['half_life_hours']:.1f} h")
        print(f"    BBB permeability: {pk['bbb_permeability']:.2f}")
        print(f"    Brain Kp: {pk['brain_partition_coefficient']:.1f}")
        print(f"    CNS MPO score: {pk['cns_mpo_score']:.1f}/6")
        print(f"    Lipinski violations: {pk['lipinski_violations']}")
