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
- Lipinski CA et al. (2001) Adv Drug Deliv Rev 46:3-26
- Waring MJ (2010) Expert Opin Drug Discov 5:235
- Di L, Kerns EH (2016) Drug-Like Properties. Academic Press.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np
import math

try:
    from pharmacology.pharmacokinetics import PKParameters
except ImportError:
    from pharmacokinetics import PKParameters


class TransporterType(Enum):
    """BBB transporter types affecting brain penetration."""
    PASSIVE = "passive"  # Lipophilic passive diffusion
    LAT1 = "lat1"  # Large neutral amino acid transporter (L-DOPA)
    OATP = "oatp"  # Organic anion transporting polypeptide
    PGP = "p-gp"  # P-glycoprotein (efflux)
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

    Can be calculated from structure using RDKit or obtained
    from databases like PubChem, DrugBank, or ChEMBL.
    """
    # Basic properties (required)
    molecular_weight: float  # g/mol (Da)
    logP: float  # Octanol-water partition coefficient

    # Polar descriptors
    psa: float = 60.0  # Polar surface area (A^2) - default for CNS drugs
    hbd: int = 2  # H-bond donors
    hba: int = 4  # H-bond acceptors

    # Ionization
    pKa_acid: Optional[float] = None
    pKa_base: Optional[float] = None

    # Structure
    rotatable_bonds: int = 5
    aromatic_rings: int = 2

    # Drug class (helps with receptor targeting)
    drug_class: DrugClass = DrugClass.UNKNOWN

    # Transporter information
    primary_transporter: TransporterType = TransporterType.PASSIVE
    pgp_substrate: bool = False  # P-gp efflux reduces brain penetration

    # Optional: Known receptor targets with Ki values (nM)
    receptor_targets: Dict[str, float] = field(default_factory=dict)


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

        # CNS drug criteria (stricter than Lipinski Ro5)
        self.cns_criteria = {
            "mw_max": 450,
            "logp_max": 5,
            "logp_min": 1,
            "psa_max": 90,
            "hbd_max": 3,
            "hba_max": 7,
        }

    def predict(self, mol: MolecularDescriptors) -> PKParameters:
        """
        Predict PK parameters from molecular descriptors.

        Args:
            mol: MolecularDescriptors object

        Returns:
            PKParameters object ready for simulation
        """
        # Predict individual parameters
        bioavailability = self._predict_bioavailability(mol)
        absorption_rate = self._predict_absorption_rate(mol)
        vd = self._predict_volume_of_distribution(mol)
        protein_binding = self._predict_protein_binding(mol)
        clearance = self._predict_clearance(mol)
        half_life = self._predict_half_life(mol, vd, clearance)
        bbb_permeability = self._predict_bbb_permeability(mol)
        brain_partition = self._predict_brain_partition(mol)

        return PKParameters(
            bioavailability=bioavailability,
            absorption_rate_constant=absorption_rate,
            volume_of_distribution_L_kg=vd,
            protein_binding_fraction=protein_binding,
            clearance_L_h_kg=clearance,
            half_life_hours=half_life,
            bbb_permeability=bbb_permeability,
            brain_partition_coefficient=brain_partition,
        )

    def _predict_bioavailability(self, mol: MolecularDescriptors) -> float:
        """
        Predict oral bioavailability based on Lipinski properties.

        Reference: Veber DF et al. (2002) J Med Chem 45:2615
        """
        # Base bioavailability
        f = 0.7

        # MW penalty
        if mol.molecular_weight > 500:
            f *= 0.8
        elif mol.molecular_weight > 600:
            f *= 0.5

        # LogP affects absorption
        if mol.logP < 0:
            f *= 0.6  # Too hydrophilic
        elif mol.logP > 5:
            f *= 0.7  # Too lipophilic (solubility issues)

        # PSA affects permeability
        if mol.psa > 140:
            f *= 0.5

        # H-bond donors reduce permeability
        if mol.hbd > 5:
            f *= 0.6

        # Rotatable bonds affect flexibility
        if mol.rotatable_bonds > 10:
            f *= 0.8

        return min(1.0, max(0.05, f))

    def _predict_absorption_rate(self, mol: MolecularDescriptors) -> float:
        """
        Predict absorption rate constant (ka, 1/h).

        Based on permeability and dissolution rate.
        """
        # Base rate (moderate absorption)
        ka = 1.0

        # Lipophilicity affects absorption
        if mol.logP > 2:
            ka *= 1.5  # Better membrane permeability
        elif mol.logP < 0:
            ka *= 0.5  # Poor permeability

        # MW affects dissolution
        if mol.molecular_weight > 400:
            ka *= 0.8

        return max(0.1, min(3.0, ka))

    def _predict_volume_of_distribution(self, mol: MolecularDescriptors) -> float:
        """
        Predict volume of distribution (L/kg).

        Reference: Smith DA et al. (2015) J Med Chem 58:5509
        """
        # Base Vd
        vd = 1.0

        # LogP strongly affects tissue distribution
        if mol.logP > 3:
            vd = 2.0 + (mol.logP - 3) * 2.0  # High lipophilicity = large Vd
        elif mol.logP > 5:
            vd = min(20.0, 10.0 + mol.logP)  # Very large Vd
        elif mol.logP < 1:
            vd = 0.5  # Hydrophilic, stays in plasma

        # Ionization affects distribution
        if mol.pKa_base and mol.pKa_base > 7.4:
            vd *= 1.5  # Basic drugs accumulate in tissues

        return max(0.2, min(30.0, vd))

    def _predict_protein_binding(self, mol: MolecularDescriptors) -> float:
        """
        Predict plasma protein binding fraction.

        Reference: Ghafourian T, Amin Z (2013) J Pharm Sci 102:595
        """
        # Base binding
        fb = 0.5

        # LogP is major determinant
        if mol.logP > 3:
            fb = 0.9 + (mol.logP - 3) * 0.02
        elif mol.logP > 2:
            fb = 0.7 + (mol.logP - 2) * 0.2
        elif mol.logP < 1:
            fb = 0.2 + mol.logP * 0.3

        # Acidic drugs bind albumin strongly
        if mol.pKa_acid and mol.pKa_acid < 5:
            fb = max(fb, 0.9)

        return min(0.99, max(0.05, fb))

    def _predict_clearance(self, mol: MolecularDescriptors) -> float:
        """
        Predict hepatic clearance (L/h/kg).

        Based on metabolic stability predictions.
        """
        # Base clearance
        cl = 0.5

        # Lipophilic drugs metabolized faster
        if mol.logP > 3:
            cl = 1.0 + (mol.logP - 3) * 0.3

        # Low MW drugs cleared faster
        if mol.molecular_weight < 300:
            cl *= 1.3

        # Drug class adjustments
        if mol.drug_class == DrugClass.ANTIDEPRESSANT_SSRI:
            cl *= 0.3  # SSRIs have low clearance
        elif mol.drug_class == DrugClass.GENERAL_ANESTHETIC:
            cl *= 2.0  # Anesthetics cleared rapidly

        return max(0.01, min(3.0, cl))

    def _predict_half_life(self, mol: MolecularDescriptors, vd: float, clearance: float) -> float:
        """
        Predict elimination half-life from Vd and clearance.

        t1/2 = 0.693 * Vd / CL
        """
        # Calculate from Vd and CL (in hours)
        vd_L = vd * self.body_weight
        cl_L_h = clearance * self.body_weight

        t_half = 0.693 * vd_L / max(0.1, cl_L_h)

        return max(0.5, min(200.0, t_half))

    def _predict_bbb_permeability(self, mol: MolecularDescriptors) -> float:
        """
        Predict blood-brain barrier permeability.

        Reference: Pajouhesh H, Lenz GR (2005) NeuroRx 2:541
        """
        # Base permeability
        p = 0.5

        # Optimal CNS drug profile
        cns_score = 0

        if mol.molecular_weight <= 450:
            cns_score += 1
        if 1 <= mol.logP <= 5:
            cns_score += 1
        if mol.psa <= 90:
            cns_score += 1
        if mol.hbd <= 3:
            cns_score += 1
        if mol.hba <= 7:
            cns_score += 1

        # Convert score to permeability
        p = 0.2 + cns_score * 0.16

        # P-gp efflux reduces brain penetration
        if mol.pgp_substrate:
            p *= 0.3

        # Active transporters can increase permeability
        if mol.primary_transporter == TransporterType.LAT1:
            p = min(0.95, p * 1.5)

        return min(0.95, max(0.05, p))

    def _predict_brain_partition(self, mol: MolecularDescriptors) -> float:
        """
        Predict brain-to-plasma partition coefficient (Kp,brain).

        Reference: Fridén M et al. (2011) Drug Metab Dispos 39:353
        """
        # Base partition
        kp = 1.0

        # LogP affects brain accumulation
        if mol.logP > 3:
            kp = 2.0 + (mol.logP - 3) * 0.5
        elif mol.logP < 1:
            kp = 0.5

        # Active transport can increase dramatically
        if mol.primary_transporter == TransporterType.LAT1:
            kp *= 5.0  # Active uptake

        # P-gp efflux reduces brain levels
        if mol.pgp_substrate:
            kp *= 0.3

        # Drug class adjustments
        if mol.drug_class == DrugClass.ANTIDEPRESSANT_SSRI:
            kp = max(kp, 5.0)  # SSRIs accumulate in brain
        elif mol.drug_class == DrugClass.DOPAMINE_PRECURSOR:
            kp *= 3.0  # LAT1 transport

        return max(0.1, min(25.0, kp))

    def is_cns_druglike(self, mol: MolecularDescriptors) -> bool:
        """
        Check if molecule meets CNS drug-likeness criteria.
        """
        violations = 0

        if mol.molecular_weight > self.cns_criteria["mw_max"]:
            violations += 1
        if mol.logP > self.cns_criteria["logp_max"]:
            violations += 1
        if mol.logP < self.cns_criteria["logp_min"]:
            violations += 1
        if mol.psa > self.cns_criteria["psa_max"]:
            violations += 1
        if mol.hbd > self.cns_criteria["hbd_max"]:
            violations += 1
        if mol.hba > self.cns_criteria["hba_max"]:
            violations += 1

        return violations <= 1


def predict_from_smiles(smiles: str, drug_class: DrugClass = DrugClass.UNKNOWN) -> PKParameters:
    """
    Predict PK parameters from SMILES string.

    Requires RDKit to be installed.

    Args:
        smiles: SMILES string of the molecule
        drug_class: Drug class for receptor targeting

    Returns:
        PKParameters object
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        raise ImportError("RDKit is required for SMILES-based prediction")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descriptors = MolecularDescriptors(
        molecular_weight=Descriptors.MolWt(mol),
        logP=Descriptors.MolLogP(mol),
        psa=Descriptors.TPSA(mol),
        hbd=rdMolDescriptors.CalcNumHBD(mol),
        hba=rdMolDescriptors.CalcNumHBA(mol),
        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        drug_class=drug_class,
    )

    predictor = MolecularPKPredictor()
    return predictor.predict(descriptors)


# Example usage and validation
if __name__ == "__main__":
    print("Testing MolecularPKPredictor\n")
    print("=" * 60)

    # Test with propofol-like molecule
    propofol_desc = MolecularDescriptors(
        molecular_weight=178.27,
        logP=3.8,
        psa=17.07,
        hbd=1,
        hba=1,
        drug_class=DrugClass.GENERAL_ANESTHETIC,
    )

    predictor = MolecularPKPredictor()
    pk = predictor.predict(propofol_desc)

    print("Propofol-like molecule prediction:")
    print(f"  Bioavailability: {pk.bioavailability:.2f}")
    print(f"  Vd: {pk.volume_of_distribution_L_kg:.1f} L/kg")
    print(f"  Protein binding: {pk.protein_binding_fraction:.2%}")
    print(f"  Clearance: {pk.clearance_L_h_kg:.2f} L/h/kg")
    print(f"  Half-life: {pk.half_life_hours:.1f} h")
    print(f"  BBB permeability: {pk.bbb_permeability:.2f}")
    print(f"  Brain Kp: {pk.brain_partition_coefficient:.1f}")
    print(f"  CNS drug-like: {predictor.is_cns_druglike(propofol_desc)}")

    print("\n" + "=" * 60)

    # Test with fluoxetine-like molecule
    fluoxetine_desc = MolecularDescriptors(
        molecular_weight=309.33,
        logP=4.05,
        psa=21.26,
        hbd=1,
        hba=2,
        drug_class=DrugClass.ANTIDEPRESSANT_SSRI,
    )

    pk2 = predictor.predict(fluoxetine_desc)

    print("Fluoxetine-like molecule prediction:")
    print(f"  Bioavailability: {pk2.bioavailability:.2f}")
    print(f"  Vd: {pk2.volume_of_distribution_L_kg:.1f} L/kg")
    print(f"  Protein binding: {pk2.protein_binding_fraction:.2%}")
    print(f"  Clearance: {pk2.clearance_L_h_kg:.2f} L/h/kg")
    print(f"  Half-life: {pk2.half_life_hours:.1f} h")
    print(f"  BBB permeability: {pk2.bbb_permeability:.2f}")
    print(f"  Brain Kp: {pk2.brain_partition_coefficient:.1f}")
    print(f"  CNS drug-like: {predictor.is_cns_druglike(fluoxetine_desc)}")
