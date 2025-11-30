"""
Mechanistic Receptor Model
==========================

First-principles biophysical model for drug-receptor interactions.
Enables prediction of effects for novel compounds and drug combinations.

Key concepts:
- BindingSite: Physical location on receptor where drug binds
- IntrinsicEfficacy: How strongly binding translates to effect (0-1)
- Cooperativity: Hill coefficient for binding (1 = no cooperativity)
- AllostericFactor: How binding at one site affects other sites

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class BindingSite(Enum):
    """Physical binding sites on GABA_A receptor."""
    BZ_SITE = "bz_site"              # Benzodiazepine site (alpha-gamma interface)
    ANESTHETIC_SITE = "anesthetic"    # Propofol/etomidate (beta subunit TM2-TM3)
    BARBITURATE_SITE = "barbiturate"  # Pentobarbital site
    NEUROSTEROID_SITE = "neurosteroid"  # Allopregnanolone site
    GABA_SITE = "gaba_site"           # Orthosteric GABA binding site
    PICROTOXIN_SITE = "picrotoxin"    # Ion channel pore (antagonist)


class EffectType(Enum):
    """Types of effects a drug can produce."""
    SEDATION = "sedation"
    ANXIOLYSIS = "anxiolysis"
    AMNESIA = "amnesia"
    MUSCLE_RELAXATION = "muscle_relaxation"
    ANTICONVULSANT = "anticonvulsant"
    ANESTHESIA = "anesthesia"


@dataclass
class DrugMolecularProfile:
    """
    Molecular and pharmacological properties of a drug.

    These are the FACTORS that determine how the molecule interacts:
    1. binding_site: WHERE it binds
    2. ki_nM: HOW TIGHTLY it binds (affinity)
    3. intrinsic_efficacy: HOW MUCH effect per binding event
    4. hill_coefficient: COOPERATIVITY of binding
    5. allosteric_factor: CROSS-SITE modulation
    """
    name: str
    binding_site: BindingSite
    ki_nM: float  # Binding affinity (lower = tighter binding)
    intrinsic_efficacy: float  # 0 = antagonist, 1 = full agonist
    hill_coefficient: float = 1.0  # Cooperativity (>1 = positive)
    allosteric_factor: float = 1.0  # Effect on other sites when bound
    effect_profile: Dict[EffectType, float] = field(default_factory=dict)
    molecular_weight: float = 300.0
    logP: float = 2.0  # Lipophilicity


# Database of drug molecular profiles
DRUG_PROFILES: Dict[str, DrugMolecularProfile] = {
    # Benzodiazepines - BZ site, moderate efficacy, no anesthesia
    "diazepam": DrugMolecularProfile(
        name="diazepam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=15.0,
        intrinsic_efficacy=0.55,  # Partial positive modulator
        hill_coefficient=1.0,
        allosteric_factor=1.2,  # Enhances GABA binding
        effect_profile={
            EffectType.ANXIOLYSIS: 0.8,
            EffectType.SEDATION: 0.5,
            EffectType.MUSCLE_RELAXATION: 0.6,
            EffectType.ANTICONVULSANT: 0.7,
            EffectType.AMNESIA: 0.4,
            EffectType.ANESTHESIA: 0.1,  # Minimal
        },
        molecular_weight=284.74,
        logP=2.82,
    ),
    "alprazolam": DrugMolecularProfile(
        name="alprazolam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=5.0,  # Higher affinity than diazepam
        intrinsic_efficacy=0.55,
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.ANXIOLYSIS: 0.9,  # Higher anxiolytic potency
            EffectType.SEDATION: 0.4,
            EffectType.MUSCLE_RELAXATION: 0.3,
            EffectType.ANTICONVULSANT: 0.5,
            EffectType.AMNESIA: 0.5,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=308.77,
        logP=2.12,
    ),
    "clonazepam": DrugMolecularProfile(
        name="clonazepam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=2.0,  # Highest affinity of BZs
        intrinsic_efficacy=0.55,
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.ANTICONVULSANT: 0.95,  # Primary indication
            EffectType.ANXIOLYSIS: 0.7,
            EffectType.SEDATION: 0.6,
            EffectType.MUSCLE_RELAXATION: 0.5,
            EffectType.AMNESIA: 0.3,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=315.71,
        logP=2.41,
    ),
    "midazolam": DrugMolecularProfile(
        name="midazolam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=6.0,
        intrinsic_efficacy=0.65,  # Slightly higher efficacy
        hill_coefficient=1.0,
        allosteric_factor=1.3,
        effect_profile={
            EffectType.AMNESIA: 0.95,  # Primary for procedural sedation
            EffectType.SEDATION: 0.8,
            EffectType.ANXIOLYSIS: 0.7,
            EffectType.ANTICONVULSANT: 0.6,
            EffectType.MUSCLE_RELAXATION: 0.3,
            EffectType.ANESTHESIA: 0.2,
        },
        molecular_weight=325.77,
        logP=3.89,
    ),
    # Anesthetics - Anesthetic site, high efficacy, full anesthesia
    "propofol": DrugMolecularProfile(
        name="propofol",
        binding_site=BindingSite.ANESTHETIC_SITE,
        ki_nM=3500.0,  # Much lower affinity but direct activation
        intrinsic_efficacy=0.95,  # Near-full positive modulator
        hill_coefficient=1.2,
        allosteric_factor=2.0,  # Strongly potentiates GABA
        effect_profile={
            EffectType.ANESTHESIA: 0.95,
            EffectType.SEDATION: 0.9,
            EffectType.AMNESIA: 0.85,
            EffectType.ANTICONVULSANT: 0.6,
            EffectType.ANXIOLYSIS: 0.3,
            EffectType.MUSCLE_RELAXATION: 0.2,
        },
        molecular_weight=178.27,
        logP=3.79,
    ),
    "etomidate": DrugMolecularProfile(
        name="etomidate",
        binding_site=BindingSite.ANESTHETIC_SITE,
        ki_nM=2000.0,
        intrinsic_efficacy=0.90,
        hill_coefficient=1.1,
        allosteric_factor=1.8,
        effect_profile={
            EffectType.ANESTHESIA: 0.90,
            EffectType.SEDATION: 0.85,
            EffectType.AMNESIA: 0.80,
            EffectType.ANTICONVULSANT: 0.5,
            EffectType.ANXIOLYSIS: 0.2,
            EffectType.MUSCLE_RELAXATION: 0.1,
        },
        molecular_weight=244.29,
        logP=2.49,
    ),
    # === NEWLY VALIDATED DRUGS ===
    "bromazepam": DrugMolecularProfile(
        name="bromazepam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=20.0,  # Similar to diazepam
        intrinsic_efficacy=0.55,  # Typical BZ efficacy
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.ANXIOLYSIS: 0.85,  # Good anxiolytic
            EffectType.SEDATION: 0.5,
            EffectType.MUSCLE_RELAXATION: 0.5,
            EffectType.ANTICONVULSANT: 0.4,
            EffectType.AMNESIA: 0.3,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=316.15,
        logP=2.05,
    ),
    "lorazepam": DrugMolecularProfile(
        name="lorazepam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=3.0,  # High affinity
        intrinsic_efficacy=0.60,
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.ANXIOLYSIS: 0.85,
            EffectType.SEDATION: 0.7,
            EffectType.AMNESIA: 0.8,  # High amnestic
            EffectType.ANTICONVULSANT: 0.9,  # Status epilepticus
            EffectType.MUSCLE_RELAXATION: 0.5,
            EffectType.ANESTHESIA: 0.15,
        },
        molecular_weight=321.16,
        logP=2.39,
    ),
    "triazolam": DrugMolecularProfile(
        name="triazolam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=4.0,
        intrinsic_efficacy=0.60,
        hill_coefficient=1.0,
        allosteric_factor=1.25,
        effect_profile={
            EffectType.SEDATION: 0.9,  # Hypnotic
            EffectType.AMNESIA: 0.85,
            EffectType.ANXIOLYSIS: 0.6,
            EffectType.ANTICONVULSANT: 0.4,
            EffectType.MUSCLE_RELAXATION: 0.3,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=343.22,
        logP=2.42,
    ),
    "temazepam": DrugMolecularProfile(
        name="temazepam",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=8.0,
        intrinsic_efficacy=0.55,
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.SEDATION: 0.85,  # Hypnotic
            EffectType.ANXIOLYSIS: 0.6,
            EffectType.AMNESIA: 0.5,
            EffectType.ANTICONVULSANT: 0.3,
            EffectType.MUSCLE_RELAXATION: 0.4,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=300.75,
        logP=2.19,
    ),
    # Volatile anesthetics
    "sevoflurane": DrugMolecularProfile(
        name="sevoflurane",
        binding_site=BindingSite.ANESTHETIC_SITE,
        ki_nM=260000.0,  # EC50 ~260 uM = very low affinity
        intrinsic_efficacy=0.90,  # High efficacy volatile
        hill_coefficient=1.0,
        allosteric_factor=2.0,
        effect_profile={
            EffectType.ANESTHESIA: 0.95,
            EffectType.SEDATION: 0.95,
            EffectType.AMNESIA: 0.9,
            EffectType.ANTICONVULSANT: 0.5,
            EffectType.MUSCLE_RELAXATION: 0.3,
            EffectType.ANXIOLYSIS: 0.2,
        },
        molecular_weight=200.05,
        logP=2.42,
    ),
    "isoflurane": DrugMolecularProfile(
        name="isoflurane",
        binding_site=BindingSite.ANESTHETIC_SITE,
        ki_nM=270000.0,  # Similar to sevoflurane
        intrinsic_efficacy=0.90,
        hill_coefficient=1.0,
        allosteric_factor=2.0,
        effect_profile={
            EffectType.ANESTHESIA: 0.95,
            EffectType.SEDATION: 0.95,
            EffectType.AMNESIA: 0.85,
            EffectType.ANTICONVULSANT: 0.4,
            EffectType.MUSCLE_RELAXATION: 0.3,
            EffectType.ANXIOLYSIS: 0.2,
        },
        molecular_weight=184.49,
        logP=2.35,
    ),
    "desflurane": DrugMolecularProfile(
        name="desflurane",
        binding_site=BindingSite.ANESTHETIC_SITE,
        ki_nM=380000.0,  # Lower potency
        intrinsic_efficacy=0.85,
        hill_coefficient=1.0,
        allosteric_factor=1.9,
        effect_profile={
            EffectType.ANESTHESIA: 0.90,
            EffectType.SEDATION: 0.90,
            EffectType.AMNESIA: 0.8,
            EffectType.ANTICONVULSANT: 0.35,
            EffectType.MUSCLE_RELAXATION: 0.25,
            EffectType.ANXIOLYSIS: 0.15,
        },
        molecular_weight=168.04,
        logP=2.08,
    ),
    # Barbiturates
    "thiopental": DrugMolecularProfile(
        name="thiopental",
        binding_site=BindingSite.BARBITURATE_SITE,
        ki_nM=5000.0,
        intrinsic_efficacy=0.95,
        hill_coefficient=1.1,
        allosteric_factor=2.2,
        effect_profile={
            EffectType.ANESTHESIA: 0.95,
            EffectType.SEDATION: 0.95,
            EffectType.AMNESIA: 0.85,
            EffectType.ANTICONVULSANT: 0.8,
            EffectType.MUSCLE_RELAXATION: 0.2,
            EffectType.ANXIOLYSIS: 0.3,
        },
        molecular_weight=242.34,
        logP=2.85,
    ),
    "phenobarbital": DrugMolecularProfile(
        name="phenobarbital",
        binding_site=BindingSite.BARBITURATE_SITE,
        ki_nM=15000.0,
        intrinsic_efficacy=0.70,
        hill_coefficient=1.0,
        allosteric_factor=1.8,
        effect_profile={
            EffectType.ANTICONVULSANT: 0.95,  # Primary use
            EffectType.SEDATION: 0.7,
            EffectType.ANESTHESIA: 0.4,
            EffectType.AMNESIA: 0.4,
            EffectType.ANXIOLYSIS: 0.5,
            EffectType.MUSCLE_RELAXATION: 0.2,
        },
        molecular_weight=232.24,
        logP=1.47,
    ),
    # Z-drugs (GABA_A BZ site but selective)
    "zolpidem": DrugMolecularProfile(
        name="zolpidem",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=10.0,  # Alpha-1 selective
        intrinsic_efficacy=0.60,
        hill_coefficient=1.0,
        allosteric_factor=1.3,
        effect_profile={
            EffectType.SEDATION: 0.95,  # Primary hypnotic
            EffectType.AMNESIA: 0.7,
            EffectType.ANXIOLYSIS: 0.3,  # Low anxiolysis (alpha-1)
            EffectType.ANTICONVULSANT: 0.2,
            EffectType.MUSCLE_RELAXATION: 0.2,
            EffectType.ANESTHESIA: 0.1,
        },
        molecular_weight=307.39,
        logP=3.0,
    ),
    "zaleplon": DrugMolecularProfile(
        name="zaleplon",
        binding_site=BindingSite.BZ_SITE,
        ki_nM=12.0,
        intrinsic_efficacy=0.55,
        hill_coefficient=1.0,
        allosteric_factor=1.2,
        effect_profile={
            EffectType.SEDATION: 0.9,  # Short-acting hypnotic
            EffectType.AMNESIA: 0.6,
            EffectType.ANXIOLYSIS: 0.25,
            EffectType.ANTICONVULSANT: 0.15,
            EffectType.MUSCLE_RELAXATION: 0.15,
            EffectType.ANESTHESIA: 0.05,
        },
        molecular_weight=305.34,
        logP=1.23,
    ),
}


class MechanisticGABAaReceptor:
    """
    First-principles GABA_A receptor model.

    Models the receptor as having multiple binding sites that can
    independently bind drugs, with allosteric interactions between sites.
    """

    def __init__(self):
        # Binding state for each site
        self.site_occupancy: Dict[BindingSite, float] = {
            site: 0.0 for site in BindingSite
        }
        self.site_drugs: Dict[BindingSite, Optional[str]] = {
            site: None for site in BindingSite
        }

        # Combined receptor state
        self.total_modulation = 1.0
        self.effect_outputs: Dict[EffectType, float] = {
            effect: 0.0 for effect in EffectType
        }

    def bind_drug(self, drug_name: str, concentration_uM: float) -> float:
        """
        Bind a drug to the receptor and calculate modulation.

        Returns modulation factor (>1 = enhancement)
        """
        if drug_name not in DRUG_PROFILES:
            raise ValueError(f"Unknown drug: {drug_name}")

        profile = DRUG_PROFILES[drug_name]
        site = profile.binding_site

        # Convert Ki to IC50 in uM
        ic50_uM = profile.ki_nM / 1000.0

        # Hill equation for binding
        hill = profile.hill_coefficient
        occupancy = (concentration_uM ** hill) / (ic50_uM ** hill + concentration_uM ** hill)

        # Update site state
        self.site_occupancy[site] = occupancy
        self.site_drugs[site] = drug_name

        # Calculate modulation based on efficacy
        modulation = 1.0 + (profile.intrinsic_efficacy * occupancy * profile.allosteric_factor)

        # Store and return
        self.total_modulation = modulation

        # Calculate effect outputs
        for effect_type, strength in profile.effect_profile.items():
            self.effect_outputs[effect_type] = strength * occupancy

        return modulation

    def bind_multiple_drugs(self, drugs: List[Tuple[str, float]]) -> float:
        """
        Bind multiple drugs and calculate combined effect.

        Args:
            drugs: List of (drug_name, concentration_uM) tuples

        Returns:
            Combined modulation factor
        """
        total_modulation = 1.0
        combined_effects: Dict[EffectType, float] = {
            effect: 0.0 for effect in EffectType
        }

        for drug_name, concentration in drugs:
            if drug_name not in DRUG_PROFILES:
                continue

            profile = DRUG_PROFILES[drug_name]
            site = profile.binding_site

            # Calculate binding
            ic50_uM = profile.ki_nM / 1000.0
            hill = profile.hill_coefficient
            occupancy = (concentration ** hill) / (ic50_uM ** hill + concentration ** hill)

            # Store occupancy
            self.site_occupancy[site] = occupancy
            self.site_drugs[site] = drug_name

            # Calculate modulation contribution
            site_mod = 1.0 + (profile.intrinsic_efficacy * occupancy * profile.allosteric_factor)

            # Check for synergy (same site = compete, different site = synergy)
            existing_sites = [p.binding_site for d, _ in drugs[:drugs.index((drug_name, concentration))]
                           if d in DRUG_PROFILES for p in [DRUG_PROFILES[d]]]

            if site in existing_sites:
                # Competition - weighted average
                total_modulation = (total_modulation + site_mod) / 2
            else:
                # Synergy - multiplicative
                total_modulation *= site_mod

            # Combine effects
            for effect_type, strength in profile.effect_profile.items():
                combined_effects[effect_type] = max(
                    combined_effects[effect_type],
                    strength * occupancy
                )

        self.total_modulation = total_modulation
        self.effect_outputs = combined_effects

        return total_modulation

    def get_effect(self, effect_type: EffectType) -> float:
        """Get the strength of a specific effect (0-1)."""
        return self.effect_outputs.get(effect_type, 0.0)

    def get_beta_power_increase(self) -> float:
        """Calculate EEG beta power increase percentage."""
        # Beta increase correlates with GABA enhancement
        return 100.0 * (self.total_modulation - 1.0)

    def get_sedation_percentage(self) -> float:
        """Calculate sedation/EEG suppression percentage."""
        # Non-linear: higher modulation = more suppression
        mod = self.total_modulation
        if mod <= 1.0:
            return 0.0

        # Sigmoid relationship
        half_max = 1.8  # Modulation at 50% sedation
        steepness = 3.0

        suppression = 100.0 / (1.0 + np.exp(-steepness * (mod - half_max)))
        return min(100.0, suppression)

    def predict_novel_drug(self,
                          ki_nM: float,
                          site: BindingSite,
                          efficacy: float,
                          concentration_uM: float) -> Dict:
        """
        Predict effect of a novel compound based on its properties.

        This is the key function for predicting unknown drugs:
        Given molecular properties, predict the receptor effect.
        """
        # Calculate binding
        ic50_uM = ki_nM / 1000.0
        occupancy = concentration_uM / (ic50_uM + concentration_uM)

        # Estimate allosteric factor based on site
        allosteric_map = {
            BindingSite.BZ_SITE: 1.2,
            BindingSite.ANESTHETIC_SITE: 2.0,
            BindingSite.BARBITURATE_SITE: 1.8,
            BindingSite.NEUROSTEROID_SITE: 1.5,
        }
        allosteric = allosteric_map.get(site, 1.0)

        # Calculate modulation
        modulation = 1.0 + (efficacy * occupancy * allosteric)

        return {
            "occupancy": occupancy,
            "modulation": modulation,
            "beta_increase_pct": 100.0 * (modulation - 1.0),
            "sedation_pct": self._estimate_sedation(modulation),
            "binding_site": site.value,
        }

    def _estimate_sedation(self, modulation: float) -> float:
        """Estimate sedation from modulation factor."""
        if modulation <= 1.0:
            return 0.0
        half_max = 1.8
        steepness = 3.0
        return min(100.0, 100.0 / (1.0 + np.exp(-steepness * (modulation - half_max))))


# =============================================================================
# DUAL MODE SYSTEM: DATABASE vs MECHANISTIC
# =============================================================================

class ModelMode(Enum):
    """Operating mode for the unified model."""
    DATABASE = "database"        # Use validated drug data
    MECHANISTIC = "mechanistic"  # First-principles prediction


class UnifiedGABAaModel:
    """
    Dual-mode GABA_A receptor model.

    DATABASE mode: Uses validated drug profiles from DRUG_PROFILES
    MECHANISTIC mode: Predicts from first principles (Ki, efficacy, site)
    """

    def __init__(self, mode: ModelMode = ModelMode.DATABASE):
        self.mode = mode
        self.receptor = MechanisticGABAaReceptor()

    def simulate(self, drug_name: str = None, concentration_uM: float = 0.0,
                 ki_nM: float = None, efficacy: float = None,
                 binding_site: BindingSite = None) -> Dict:
        """
        Simulate drug effect based on mode.

        DATABASE mode requires: drug_name, concentration_uM
        MECHANISTIC mode requires: ki_nM, efficacy, binding_site, concentration_uM
        """
        if self.mode == ModelMode.DATABASE:
            if drug_name is None:
                raise ValueError("DATABASE mode requires drug_name")
            if drug_name not in DRUG_PROFILES:
                raise ValueError(f"Drug '{drug_name}' not in database. Use MECHANISTIC mode.")

            mod = self.receptor.bind_drug(drug_name, concentration_uM)
            profile = DRUG_PROFILES[drug_name]

            return {
                "mode": "DATABASE",
                "drug": drug_name,
                "concentration_uM": concentration_uM,
                "binding_site": profile.binding_site.value,
                "ki_nM": profile.ki_nM,
                "efficacy": profile.intrinsic_efficacy,
                "occupancy": self.receptor.site_occupancy[profile.binding_site],
                "modulation": mod,
                "beta_increase_pct": self.receptor.get_beta_power_increase(),
                "sedation_pct": self.receptor.get_sedation_percentage(),
                "effects": {k.value: v for k, v in self.receptor.effect_outputs.items()},
            }

        else:  # MECHANISTIC mode
            if None in (ki_nM, efficacy, binding_site):
                raise ValueError("MECHANISTIC mode requires ki_nM, efficacy, binding_site")

            result = self.receptor.predict_novel_drug(ki_nM, binding_site, efficacy, concentration_uM)

            return {
                "mode": "MECHANISTIC",
                "drug": "novel_compound",
                "concentration_uM": concentration_uM,
                "binding_site": binding_site.value,
                "ki_nM": ki_nM,
                "efficacy": efficacy,
                **result,
            }

    def simulate_interaction(self, drugs: List[Tuple[str, float]]) -> Dict:
        """
        Simulate drug-drug interaction.

        Args:
            drugs: List of (drug_name, concentration_uM) tuples

        Returns:
            Combined effect with synergy/competition analysis
        """
        mod = self.receptor.bind_multiple_drugs(drugs)

        # Analyze interaction type
        sites_used = []
        for drug_name, _ in drugs:
            if drug_name in DRUG_PROFILES:
                sites_used.append(DRUG_PROFILES[drug_name].binding_site)

        unique_sites = len(set(sites_used))
        interaction_type = "synergy" if unique_sites > 1 else "competition"

        return {
            "mode": "INTERACTION",
            "drugs": drugs,
            "sites_involved": [s.value for s in set(sites_used)],
            "interaction_type": interaction_type,
            "combined_modulation": mod,
            "beta_increase_pct": self.receptor.get_beta_power_increase(),
            "sedation_pct": self.receptor.get_sedation_percentage(),
            "effects": {k.value: v for k, v in self.receptor.effect_outputs.items()},
        }


# =============================================================================
# REVERSE ENGINEERING: Infer drugs from observed effects
# =============================================================================

class EffectReverseEngineer:
    """
    Given observed effects, infer what drugs could cause them.

    Use cases:
    - Unknown intoxication diagnosis
    - Forensic toxicology
    - Drug interaction investigation
    """

    def __init__(self):
        self.drug_profiles = DRUG_PROFILES

    def infer_from_sedation(self, sedation_pct: float, tolerance: float = 10.0) -> List[Dict]:
        """
        Given observed sedation %, find drugs that could cause it.

        Returns list of candidate drugs with estimated concentrations.
        """
        candidates = []

        for drug_name, profile in self.drug_profiles.items():
            # Calculate what concentration would give this sedation
            receptor = MechanisticGABAaReceptor()

            # Binary search for concentration
            conc_low, conc_high = 0.001, 100.0
            best_conc = None
            best_error = float('inf')

            for _ in range(20):  # Binary search iterations
                conc_mid = (conc_low + conc_high) / 2
                receptor = MechanisticGABAaReceptor()
                receptor.bind_drug(drug_name, conc_mid)
                predicted_sedation = receptor.get_sedation_percentage()

                error = abs(predicted_sedation - sedation_pct)
                if error < best_error:
                    best_error = error
                    best_conc = conc_mid

                if predicted_sedation < sedation_pct:
                    conc_low = conc_mid
                else:
                    conc_high = conc_mid

            if best_error <= tolerance:
                candidates.append({
                    "drug": drug_name,
                    "estimated_concentration_uM": best_conc,
                    "predicted_sedation_pct": sedation_pct,
                    "error_pct": best_error,
                    "binding_site": profile.binding_site.value,
                    "clinical_plausibility": "high" if 0.01 < best_conc < 50 else "low",
                })

        # Sort by clinical plausibility
        candidates.sort(key=lambda x: (x["clinical_plausibility"] != "high", x["error_pct"]))
        return candidates

    def infer_from_beta_increase(self, beta_increase_pct: float, tolerance: float = 10.0) -> List[Dict]:
        """
        Given observed EEG beta increase %, find drugs that could cause it.
        """
        candidates = []

        for drug_name, profile in self.drug_profiles.items():
            receptor = MechanisticGABAaReceptor()

            # Binary search for concentration
            conc_low, conc_high = 0.001, 100.0
            best_conc = None
            best_error = float('inf')

            for _ in range(20):
                conc_mid = (conc_low + conc_high) / 2
                receptor = MechanisticGABAaReceptor()
                receptor.bind_drug(drug_name, conc_mid)
                predicted_beta = receptor.get_beta_power_increase()

                error = abs(predicted_beta - beta_increase_pct)
                if error < best_error:
                    best_error = error
                    best_conc = conc_mid

                if predicted_beta < beta_increase_pct:
                    conc_low = conc_mid
                else:
                    conc_high = conc_mid

            if best_error <= tolerance:
                candidates.append({
                    "drug": drug_name,
                    "estimated_concentration_uM": best_conc,
                    "predicted_beta_increase_pct": beta_increase_pct,
                    "error_pct": best_error,
                    "binding_site": profile.binding_site.value,
                })

        candidates.sort(key=lambda x: x["error_pct"])
        return candidates

    def infer_from_effect_pattern(self, effects: Dict[str, float]) -> List[Dict]:
        """
        Given a pattern of effects, find the best matching drug(s).

        Args:
            effects: Dict like {"sedation": 0.6, "anxiolysis": 0.8, "amnesia": 0.3}
        """
        candidates = []

        for drug_name, profile in self.drug_profiles.items():
            # Calculate match score
            match_score = 0.0
            total_weight = 0.0

            for effect_name, observed_value in effects.items():
                try:
                    effect_type = EffectType(effect_name)
                    if effect_type in profile.effect_profile:
                        drug_effect = profile.effect_profile[effect_type]
                        # Score based on how close the effect is
                        match = 1.0 - abs(drug_effect - observed_value)
                        match_score += max(0, match)
                        total_weight += 1.0
                except ValueError:
                    continue

            if total_weight > 0:
                avg_match = match_score / total_weight
                if avg_match > 0.5:  # At least 50% match
                    candidates.append({
                        "drug": drug_name,
                        "match_score": avg_match,
                        "binding_site": profile.binding_site.value,
                        "drug_effect_profile": {k.value: v for k, v in profile.effect_profile.items()},
                    })

        candidates.sort(key=lambda x: -x["match_score"])
        return candidates


def demo_mechanistic_model():
    """Demonstrate the mechanistic model capabilities."""
    print("\n" + "=" * 80)
    print("MECHANISTIC RECEPTOR MODEL DEMO")
    print("=" * 80)

    receptor = MechanisticGABAaReceptor()

    # Test known drugs
    print("\n1. Known Drug Effects (single drug):")
    print("-" * 60)

    test_cases = [
        ("diazepam", 0.5),    # Clinical anxiolytic dose
        ("alprazolam", 0.05),  # Lower dose due to higher affinity
        ("propofol", 5.0),    # Anesthetic dose
    ]

    for drug, conc in test_cases:
        receptor = MechanisticGABAaReceptor()  # Reset
        mod = receptor.bind_drug(drug, conc)
        profile = DRUG_PROFILES[drug]

        print(f"\n{drug.upper()} at {conc} uM:")
        print(f"  Binding site: {profile.binding_site.value}")
        print(f"  Ki: {profile.ki_nM} nM")
        print(f"  Intrinsic efficacy: {profile.intrinsic_efficacy}")
        print(f"  Occupancy: {receptor.site_occupancy[profile.binding_site]:.1%}")
        print(f"  Modulation: {mod:.2f}x")
        print(f"  Beta increase: {receptor.get_beta_power_increase():.1f}%")
        print(f"  Sedation: {receptor.get_sedation_percentage():.1f}%")

    # Test drug combination
    print("\n\n2. Drug Combination (Synergy at Different Sites):")
    print("-" * 60)

    receptor = MechanisticGABAaReceptor()
    combo = [("diazepam", 0.2), ("propofol", 2.0)]
    mod = receptor.bind_multiple_drugs(combo)

    print(f"\nDiazepam 0.2 uM + Propofol 2.0 uM:")
    print(f"  Combined modulation: {mod:.2f}x")
    print(f"  Beta increase: {receptor.get_beta_power_increase():.1f}%")
    print(f"  Sedation: {receptor.get_sedation_percentage():.1f}%")
    print(f"  (Note: Different sites -> multiplicative synergy)")

    # Test novel drug prediction
    print("\n\n3. Novel Drug Prediction:")
    print("-" * 60)

    receptor = MechanisticGABAaReceptor()
    prediction = receptor.predict_novel_drug(
        ki_nM=10.0,  # High affinity
        site=BindingSite.BZ_SITE,
        efficacy=0.8,  # Higher efficacy than typical BZ
        concentration_uM=0.1
    )

    print(f"\nHypothetical novel BZ (Ki=10nM, efficacy=0.8):")
    for key, value in prediction.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_mechanistic_model()
