"""
Mechanistic Receptor Models - First Principles Pharmacodynamics
================================================================

Models receptor binding based on biophysical mechanisms, not drug-specific
parameters. This allows prediction for ANY drug based on its molecular
properties and mechanism of action.

Architecture:
- Binding sites have FIXED biophysical properties
- Drugs have molecular properties that determine interaction
- Effect emerges from mechanism, not calibration

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Literal
from enum import Enum


class BindingSite(Enum):
    """GABA-A receptor binding sites with distinct mechanisms"""
    BZ_SITE = "bz_site"  # Benzodiazepine site (α-γ interface)
    ANESTHETIC_SITE = "anesthetic_site"  # β subunit (propofol, etomidate)
    BARBITURATE_SITE = "barbiturate_site"  # Distinct from BZ
    NEUROSTEROID_SITE = "neurosteroid_site"  # Endogenous modulation
    GABA_SITE = "gaba_site"  # Orthosteric agonist site


class DrugClass(Enum):
    """Pharmacological drug classes with shared mechanisms"""
    BENZODIAZEPINE = "benzodiazepine"
    IV_ANESTHETIC = "iv_anesthetic"
    VOLATILE_ANESTHETIC = "volatile_anesthetic"
    BARBITURATE = "barbiturate"
    OPIOID = "opioid"
    ANTIPSYCHOTIC = "antipsychotic"
    SSRI = "ssri"
    NMDA_ANTAGONIST = "nmda_antagonist"


@dataclass
class DrugMolecularProperties:
    """
    Molecular properties that determine pharmacodynamics.

    These are intrinsic drug properties, NOT receptor-specific calibrations.
    From these, we CALCULATE the receptor interaction.
    """
    name: str
    molecular_weight: float  # g/mol
    logP: float  # Lipophilicity (octanol-water partition)
    pKa: float  # Acid dissociation constant
    drug_class: DrugClass
    binding_site: BindingSite

    # These CAN be measured or predicted from QSAR
    Ki_nM: float  # Binding affinity (literature or predicted)
    intrinsic_efficacy: float  # 0-1, how much it activates when bound

    # Optional: for volatile anesthetics
    mac: Optional[float] = None  # Minimum Alveolar Concentration (%)
    blood_gas_partition: Optional[float] = None


# =============================================================================
# BINDING SITE BIOPHYSICS (FIXED - these are receptor properties, not drug)
# =============================================================================

@dataclass
class BindingSiteProperties:
    """
    Biophysical properties of a receptor binding site.
    These are FIXED for the receptor, independent of drug.
    """
    name: str
    max_modulation: float  # Maximum effect when fully occupied
    hill_coefficient: float  # Cooperativity
    baseline_activity: float  # Constitutive activity

    # Kinetics
    typical_koff_per_s: float  # Typical dissociation rate

    # Mechanism type
    is_allosteric: bool  # True = modulates GABA, False = direct action
    requires_gaba: bool  # True = no effect without GABA


# GABA-A receptor binding sites with their biophysical properties
GABA_A_BINDING_SITES = {
    BindingSite.BZ_SITE: BindingSiteProperties(
        name="Benzodiazepine Site (α-γ interface)",
        max_modulation=1.5,  # BZs enhance GABA ~50-150%
        hill_coefficient=1.0,
        baseline_activity=0.0,
        typical_koff_per_s=0.1,
        is_allosteric=True,
        requires_gaba=True,  # BZs need GABA to work
    ),
    BindingSite.ANESTHETIC_SITE: BindingSiteProperties(
        name="Anesthetic Site (β subunit)",
        max_modulation=4.0,  # Anesthetics can cause direct activation
        hill_coefficient=1.2,
        baseline_activity=0.0,
        typical_koff_per_s=1.0,
        is_allosteric=False,  # Can directly open channel at high conc
        requires_gaba=False,  # Direct effect at high doses
    ),
    BindingSite.BARBITURATE_SITE: BindingSiteProperties(
        name="Barbiturate Site",
        max_modulation=5.0,  # Very potent direct activation
        hill_coefficient=1.5,
        baseline_activity=0.0,
        typical_koff_per_s=0.5,
        is_allosteric=False,
        requires_gaba=False,
    ),
}


# =============================================================================
# MECHANISTIC GABA-A RECEPTOR MODEL
# =============================================================================

class GABAaMechanistic:
    """
    Mechanistic GABA-A receptor model.

    Effect is calculated from:
    1. Drug molecular properties (Ki, efficacy)
    2. Binding site biophysics (max_modulation, cooperativity)
    3. NOT from drug-specific calibration

    This allows prediction for ANY drug with known properties.
    """

    def __init__(self):
        # GABA parameters (endogenous ligand)
        self.gaba_ec50_uM = 10.0
        self.gaba_hill = 1.5

        # Baseline state
        self.gaba_concentration_uM = 1.0  # Tonic GABA
        self.modulation_factor = 1.0
        self.drug_occupancy = 0.0

        # Multi-drug support
        self.bound_drugs: Dict[str, float] = {}

    def bind_drug_mechanistic(
        self,
        concentration_uM: float,
        drug_props: DrugMolecularProperties,
        gaba_concentration_uM: float = 1.0
    ) -> float:
        """
        Calculate drug effect from molecular properties and site biophysics.

        This is the CORE mechanistic model:
        1. Calculate occupancy from Ki and concentration (Hill equation)
        2. Get site-specific max_modulation
        3. Apply intrinsic efficacy
        4. Check GABA dependency

        Args:
            concentration_uM: Drug concentration at receptor (μM)
            drug_props: Molecular properties of the drug
            gaba_concentration_uM: Local GABA concentration

        Returns:
            Modulation factor (1.0 = no effect, >1 = enhancement)
        """
        # Get binding site properties
        site_props = GABA_A_BINDING_SITES.get(drug_props.binding_site)
        if site_props is None:
            raise ValueError(f"Unknown binding site: {drug_props.binding_site}")

        # Convert Ki to uM for consistent units
        Ki_uM = drug_props.Ki_nM / 1000.0

        # Hill equation for fractional occupancy
        if concentration_uM > 0:
            self.drug_occupancy = (concentration_uM ** site_props.hill_coefficient) / (
                Ki_uM ** site_props.hill_coefficient +
                concentration_uM ** site_props.hill_coefficient
            )
        else:
            self.drug_occupancy = 0.0

        # Store for multi-drug calculations
        self.bound_drugs[drug_props.name] = self.drug_occupancy

        # Calculate modulation
        # Effect = max_modulation × occupancy × intrinsic_efficacy
        raw_modulation = site_props.max_modulation * self.drug_occupancy * drug_props.intrinsic_efficacy

        # GABA dependency for allosteric modulators
        if site_props.requires_gaba:
            # BZs need GABA - effect scales with GABA occupancy
            # NOTE: Clinical effects (EEG, sedation) are measured during normal
            # synaptic activity where GABA levels are MUCH higher than tonic.
            # Average synaptic GABA during activity: ~50-100 uM (peak ~mM)
            # We use "effective GABA" = tonic + synaptic activity contribution
            effective_gaba = gaba_concentration_uM + 20.0  # Synaptic activity baseline
            gaba_occupancy = effective_gaba / (self.gaba_ec50_uM + effective_gaba)
            raw_modulation *= gaba_occupancy

        # Final modulation factor (1.0 = baseline)
        self.modulation_factor = 1.0 + raw_modulation

        return self.modulation_factor

    def get_clinical_effect(self, effect_type: str = "suppression") -> float:
        """
        Convert receptor modulation to clinical effect.

        The relationship between modulation and effect depends on:
        - Brain region affected
        - Network dynamics
        - Clinical endpoint measured

        Args:
            effect_type: "suppression" (EEG), "beta_power", "sedation"

        Returns:
            Clinical effect percentage
        """
        if effect_type == "suppression":
            # EEG suppression (for anesthetics)
            # Modulation ~2.5 → ~60% suppression
            suppression = 100.0 * (1.0 - 1.0 / self.modulation_factor)
            return min(95.0, suppression)

        elif effect_type == "beta_power":
            # Beta band increase (for BZs)
            # More linear relationship
            beta = (self.modulation_factor - 1.0) * 100.0
            return min(80.0, beta)

        elif effect_type == "sedation":
            # Sedation score (Ramsay-like)
            # Emax model for ARAS inhibition
            effect = self.modulation_factor - 1.0
            ec50_effect = 0.8
            emax = 95.0
            sedation = emax * effect / (ec50_effect + effect) if effect > 0 else 0
            return max(0.0, min(95.0, sedation))

        else:
            return (self.modulation_factor - 1.0) * 50.0


# =============================================================================
# DRUG PROPERTY DATABASE (Minimal - just molecular properties)
# =============================================================================

DRUG_PROPERTIES = {
    # BENZODIAZEPINES - all use BZ site
    "diazepam": DrugMolecularProperties(
        name="diazepam",
        molecular_weight=284.74,
        logP=2.82,
        pKa=3.4,
        drug_class=DrugClass.BENZODIAZEPINE,
        binding_site=BindingSite.BZ_SITE,
        Ki_nM=15.0,  # Literature: ~15 nM at BZ site
        intrinsic_efficacy=0.6,  # Partial efficacy for sedation
    ),
    "midazolam": DrugMolecularProperties(
        name="midazolam",
        molecular_weight=325.77,
        logP=3.89,
        pKa=6.2,  # Water soluble at low pH
        drug_class=DrugClass.BENZODIAZEPINE,
        binding_site=BindingSite.BZ_SITE,
        Ki_nM=20.0,  # Similar to diazepam
        intrinsic_efficacy=0.85,  # Higher efficacy for sedation
    ),
    "lorazepam": DrugMolecularProperties(
        name="lorazepam",
        molecular_weight=321.16,
        logP=2.39,
        pKa=1.3,
        drug_class=DrugClass.BENZODIAZEPINE,
        binding_site=BindingSite.BZ_SITE,
        Ki_nM=12.0,  # Very high affinity
        intrinsic_efficacy=0.7,  # Moderate-high efficacy
    ),
    "clonazepam": DrugMolecularProperties(
        name="clonazepam",
        molecular_weight=315.71,
        logP=2.41,
        pKa=1.5,
        drug_class=DrugClass.BENZODIAZEPINE,
        binding_site=BindingSite.BZ_SITE,
        Ki_nM=0.3,  # Extremely high affinity
        intrinsic_efficacy=0.65,  # Anticonvulsant focus
    ),

    # IV ANESTHETICS - use anesthetic site
    "propofol": DrugMolecularProperties(
        name="propofol",
        molecular_weight=178.27,
        logP=3.79,
        pKa=11.1,  # Weakly acidic phenol
        drug_class=DrugClass.IV_ANESTHETIC,
        binding_site=BindingSite.ANESTHETIC_SITE,
        Ki_nM=3500.0,  # EC50 ~3.5 μM = 3500 nM
        intrinsic_efficacy=0.85,
    ),

    # VOLATILE ANESTHETICS - also use anesthetic site
    "sevoflurane": DrugMolecularProperties(
        name="sevoflurane",
        molecular_weight=200.05,
        logP=1.85,  # Less lipophilic than other volatiles
        pKa=None,  # Not applicable
        drug_class=DrugClass.VOLATILE_ANESTHETIC,
        binding_site=BindingSite.ANESTHETIC_SITE,
        Ki_nM=500000.0,  # Very high EC50 in solution (~500 μM)
        intrinsic_efficacy=0.80,
        mac=2.0,  # MAC 2.0%
        blood_gas_partition=0.65,
    ),
    "isoflurane": DrugMolecularProperties(
        name="isoflurane",
        molecular_weight=184.49,
        logP=2.06,
        pKa=None,
        drug_class=DrugClass.VOLATILE_ANESTHETIC,
        binding_site=BindingSite.ANESTHETIC_SITE,
        Ki_nM=400000.0,
        intrinsic_efficacy=0.82,
        mac=1.15,
        blood_gas_partition=1.4,
    ),
}


def get_drug_properties(drug_name: str) -> DrugMolecularProperties:
    """Get molecular properties for a drug."""
    drug_lower = drug_name.lower()
    if drug_lower not in DRUG_PROPERTIES:
        raise ValueError(f"Unknown drug: {drug_name}. Available: {list(DRUG_PROPERTIES.keys())}")
    return DRUG_PROPERTIES[drug_lower]


# =============================================================================
# QSAR FUNCTIONS - Predict Ki from molecular properties
# =============================================================================

def predict_bz_Ki_from_structure(logP: float, mw: float) -> float:
    """
    Predict benzodiazepine Ki from molecular properties.

    Simple QSAR based on:
    - Lipophilicity (logP) affects membrane partitioning
    - Size (MW) affects binding pocket fit

    This is a simplified model - real QSAR uses more descriptors.

    Args:
        logP: Octanol-water partition coefficient
        mw: Molecular weight (g/mol)

    Returns:
        Predicted Ki in nM
    """
    # Empirical relationship for BZ site
    # Higher logP → better binding (lower Ki)
    # Optimal MW around 280-330

    # Base Ki ~ 50 nM for average BZ
    base_Ki = 50.0

    # logP correction: every 0.5 increase in logP → 2x better binding
    logP_factor = 2.0 ** ((2.5 - logP) / 0.5)

    # MW correction: optimal around 300, deviation reduces affinity
    mw_optimal = 300.0
    mw_deviation = abs(mw - mw_optimal) / 50.0
    mw_factor = 1.0 + 0.5 * mw_deviation

    predicted_Ki = base_Ki * logP_factor * mw_factor

    return max(0.1, min(1000.0, predicted_Ki))  # Clamp to reasonable range


def predict_anesthetic_EC50(logP: float, mw: float) -> float:
    """
    Predict anesthetic EC50 from molecular properties.

    Meyer-Overton correlation: anesthetic potency correlates with lipophilicity.

    Returns:
        Predicted EC50 in nM (for brain concentration)
    """
    # Meyer-Overton: potency ∝ lipophilicity
    # logP of 3 → EC50 ~ 5000 nM (5 μM)
    base_EC50 = 5000.0

    # Every unit increase in logP → 3x more potent
    logP_factor = 3.0 ** (3.0 - logP)

    predicted_EC50 = base_EC50 * logP_factor

    return max(100.0, min(1000000.0, predicted_EC50))


# =============================================================================
# PREDICTION MODE - Dual Architecture
# =============================================================================

class PredictionMode(Enum):
    """
    Two complementary prediction approaches:

    DATABASE: Uses calibrated drug-specific parameters
        - Best for: Known drugs, cross-drug interactions, dose optimization
        - Example: "What happens if patient takes diazepam + alcohol?"

    MECHANISTIC: Uses first-principles biophysics only
        - Best for: Novel compounds, behavior prediction, reverse engineering
        - Example: "What drug profile caused these EEG changes?"
    """
    DATABASE = "database"
    MECHANISTIC = "mechanistic"


# =============================================================================
# UNIFIED RECEPTOR MODEL - Supports both modes
# =============================================================================

class UnifiedGABAaModel:
    """
    Unified GABA-A receptor model supporting both prediction modes.

    DATABASE mode:
        - Uses drug-specific calibrated parameters (synapse_models.py)
        - Accurate for known drugs and their interactions
        - Supports multi-drug binding and potentiation

    MECHANISTIC mode:
        - Uses only molecular properties + receptor biophysics
        - Can predict effects of ANY compound with known Ki/efficacy
        - Enables reverse engineering (effect → probable cause)
    """

    def __init__(self, mode: PredictionMode = PredictionMode.MECHANISTIC):
        self.mode = mode
        self.mechanistic_model = GABAaMechanistic()

        # Multi-drug state tracking
        self.bound_drugs: Dict[str, Dict] = {}  # drug_name -> {conc, occupancy, site}
        self.total_modulation = 1.0

        # For reverse engineering
        self.effect_history: list = []

    def bind_drug(
        self,
        drug_name: str,
        concentration_uM: float,
        gaba_concentration_uM: float = 1.0,
        custom_props: Optional[DrugMolecularProperties] = None
    ) -> Dict:
        """
        Bind a drug to the receptor using selected mode.

        Args:
            drug_name: Name of the drug (must be in database for DATABASE mode)
            concentration_uM: Drug concentration at receptor
            gaba_concentration_uM: Local GABA concentration
            custom_props: Optional custom properties (for novel compounds)

        Returns:
            Dict with binding results and predicted effects
        """
        if self.mode == PredictionMode.MECHANISTIC:
            return self._bind_mechanistic(drug_name, concentration_uM,
                                          gaba_concentration_uM, custom_props)
        else:
            return self._bind_database(drug_name, concentration_uM,
                                       gaba_concentration_uM)

    def _bind_mechanistic(
        self,
        drug_name: str,
        concentration_uM: float,
        gaba_concentration_uM: float,
        custom_props: Optional[DrugMolecularProperties]
    ) -> Dict:
        """Mechanistic binding using molecular properties."""
        # Get or create drug properties
        if custom_props:
            props = custom_props
        elif drug_name.lower() in DRUG_PROPERTIES:
            props = DRUG_PROPERTIES[drug_name.lower()]
        else:
            raise ValueError(f"Unknown drug '{drug_name}'. Provide custom_props for novel compounds.")

        # Calculate binding using mechanistic model
        modulation = self.mechanistic_model.bind_drug_mechanistic(
            concentration_uM, props, gaba_concentration_uM
        )

        # Store state
        self.bound_drugs[drug_name] = {
            "concentration_uM": concentration_uM,
            "occupancy": self.mechanistic_model.drug_occupancy,
            "site": props.binding_site.value,
            "modulation": modulation,
            "props": props
        }

        # Calculate cumulative modulation from all bound drugs
        self._update_total_modulation()

        return {
            "drug": drug_name,
            "mode": "mechanistic",
            "occupancy": self.mechanistic_model.drug_occupancy,
            "modulation": modulation,
            "total_modulation": self.total_modulation,
            "beta_power_pct": self.get_effect("beta_power"),
            "sedation_pct": self.get_effect("sedation"),
            "suppression_pct": self.get_effect("suppression")
        }

    def _bind_database(
        self,
        drug_name: str,
        concentration_uM: float,
        gaba_concentration_uM: float
    ) -> Dict:
        """
        Database binding using calibrated parameters.
        This calls the original GABAaReceptor from synapse_models.
        """
        # Import here to avoid circular dependency
        from simulation.synapse_models import GABAaReceptor

        receptor = GABAaReceptor()
        receptor.bind_drug(concentration_uM, drug_type=drug_name)

        self.bound_drugs[drug_name] = {
            "concentration_uM": concentration_uM,
            "occupancy": receptor.drug_bound_fraction,
            "modulation": receptor.modulation_factor
        }

        self.total_modulation = receptor.modulation_factor

        return {
            "drug": drug_name,
            "mode": "database",
            "occupancy": receptor.drug_bound_fraction,
            "modulation": receptor.modulation_factor,
            "beta_power_pct": receptor.get_beta_power_increase_percent(),
            "sedation_pct": receptor.get_sedation_percentage(),
            "suppression_pct": receptor.get_suppression_percentage()
        }

    def _update_total_modulation(self):
        """Calculate cumulative effect from all bound drugs."""
        # Drugs at SAME site compete (Bliss independence model)
        # Drugs at DIFFERENT sites multiply (synergy)

        site_modulations: Dict[str, float] = {}

        for drug_data in self.bound_drugs.values():
            site = drug_data.get("site", "unknown")
            mod = drug_data.get("modulation", 1.0)

            if site in site_modulations:
                # Same site: competitive binding - take maximum
                site_modulations[site] = max(site_modulations[site], mod)
            else:
                site_modulations[site] = mod

        # Different sites multiply (but subtract 1, add back)
        # E.g., 1.5 and 2.0 → (0.5 + 1.0) + 1 = 2.5, not 3.0
        total = 1.0
        for mod in site_modulations.values():
            total += (mod - 1.0)

        self.total_modulation = total

    def bind_multiple_drugs(
        self,
        drug_concentrations: Dict[str, float],
        gaba_concentration_uM: float = 1.0
    ) -> Dict:
        """
        Bind multiple drugs simultaneously (for interaction prediction).

        Args:
            drug_concentrations: Dict of drug_name -> concentration_uM

        Returns:
            Combined effect prediction including interactions
        """
        results = {}

        for drug_name, conc in drug_concentrations.items():
            results[drug_name] = self.bind_drug(
                drug_name, conc, gaba_concentration_uM
            )

        # Calculate interaction effects
        interaction_type = self._classify_interaction()

        return {
            "individual_results": results,
            "total_modulation": self.total_modulation,
            "interaction_type": interaction_type,
            "combined_effects": {
                "beta_power_pct": self.get_effect("beta_power"),
                "sedation_pct": self.get_effect("sedation"),
                "suppression_pct": self.get_effect("suppression")
            }
        }

    def _classify_interaction(self) -> str:
        """Classify drug-drug interaction type."""
        if len(self.bound_drugs) < 2:
            return "single_drug"

        # Check if drugs are at same or different sites
        sites = set(d.get("site") for d in self.bound_drugs.values())

        if len(sites) == 1:
            return "competitive"  # Same site - may reduce efficacy
        else:
            return "synergistic"  # Different sites - effects multiply

    def get_effect(self, effect_type: str) -> float:
        """Get clinical effect from current modulation state."""
        # Use total modulation for multi-drug scenarios
        self.mechanistic_model.modulation_factor = self.total_modulation
        return self.mechanistic_model.get_clinical_effect(effect_type)

    def clear_drugs(self):
        """Clear all bound drugs (reset receptor)."""
        self.bound_drugs = {}
        self.total_modulation = 1.0
        self.mechanistic_model = GABAaMechanistic()


# =============================================================================
# REVERSE ENGINEERING - Infer drug from observed effects
# =============================================================================

class EffectReverseEngineer:
    """
    Reverse engineering: Given observed effects, infer probable cause.

    This is the key capability enabled by mechanistic modeling:
    - Input: Observed clinical effects (EEG pattern, sedation level, etc.)
    - Output: Most likely drug class, binding site, and concentration range

    Use cases:
    - Unknown intoxication diagnosis
    - Drug effect attribution
    - Forensic analysis
    """

    def __init__(self):
        self.model = UnifiedGABAaModel(mode=PredictionMode.MECHANISTIC)

    def infer_from_effects(
        self,
        observed_effects: Dict[str, float],
        tolerance: float = 0.15
    ) -> list:
        """
        Infer probable drug(s) from observed clinical effects.

        Args:
            observed_effects: Dict of effect_type -> observed_value
                e.g., {"beta_power_pct": 45.0, "sedation_pct": 30.0}
            tolerance: Acceptable error margin (0.15 = 15%)

        Returns:
            List of candidate drugs with estimated concentrations
        """
        candidates = []

        for drug_name, props in DRUG_PROPERTIES.items():
            # Try range of concentrations
            for conc_factor in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                # Estimate typical therapeutic concentration
                base_conc = props.Ki_nM / 1000.0  # Ki in uM as starting point
                test_conc = base_conc * conc_factor

                # Predict effects at this concentration
                self.model.clear_drugs()
                result = self.model.bind_drug(drug_name, test_conc)

                # Check match against observed effects
                match_score = self._calculate_match_score(observed_effects, result)

                if match_score > (1 - tolerance):
                    candidates.append({
                        "drug": drug_name,
                        "drug_class": props.drug_class.value,
                        "binding_site": props.binding_site.value,
                        "estimated_concentration_uM": test_conc,
                        "match_score": match_score,
                        "predicted_effects": {
                            "beta_power_pct": result["beta_power_pct"],
                            "sedation_pct": result["sedation_pct"],
                            "suppression_pct": result["suppression_pct"]
                        }
                    })

        # Sort by match score
        candidates.sort(key=lambda x: x["match_score"], reverse=True)

        return candidates[:5]  # Top 5 candidates

    def _calculate_match_score(
        self,
        observed: Dict[str, float],
        predicted: Dict
    ) -> float:
        """Calculate how well predicted effects match observed."""
        scores = []

        for effect_type, observed_value in observed.items():
            predicted_key = f"{effect_type}"
            if predicted_key in predicted:
                predicted_value = predicted[predicted_key]
                if observed_value > 0:
                    error = abs(predicted_value - observed_value) / observed_value
                    score = max(0, 1 - error)
                    scores.append(score)

        return np.mean(scores) if scores else 0.0

    def infer_binding_site(
        self,
        observed_effects: Dict[str, float]
    ) -> Dict:
        """
        Infer which binding site is most likely affected.

        Different binding sites produce characteristic effect patterns:
        - BZ site: High beta power, moderate sedation, low suppression
        - Anesthetic site: High suppression, high sedation, moderate beta
        - Barbiturate site: Very high suppression, very high sedation
        """
        patterns = {
            "BZ_SITE": {
                "beta_power_pct": (30, 80),
                "sedation_pct": (20, 50),
                "suppression_pct": (5, 30)
            },
            "ANESTHETIC_SITE": {
                "beta_power_pct": (10, 40),
                "sedation_pct": (50, 95),
                "suppression_pct": (40, 80)
            },
            "BARBITURATE_SITE": {
                "beta_power_pct": (5, 30),
                "sedation_pct": (70, 99),
                "suppression_pct": (60, 95)
            }
        }

        scores = {}
        for site, expected in patterns.items():
            score = 0
            total = 0
            for effect_type, (low, high) in expected.items():
                if effect_type in observed_effects:
                    observed = observed_effects[effect_type]
                    if low <= observed <= high:
                        score += 1
                    total += 1
            scores[site] = score / total if total > 0 else 0

        best_site = max(scores, key=scores.get)

        return {
            "most_likely_site": best_site,
            "confidence_scores": scores,
            "interpretation": self._interpret_site(best_site)
        }

    def _interpret_site(self, site: str) -> str:
        """Provide clinical interpretation of binding site."""
        interpretations = {
            "BZ_SITE": "Probable benzodiazepine or Z-drug exposure. "
                       "Look for anxiolytic and anticonvulsant effects.",
            "ANESTHETIC_SITE": "Probable propofol, etomidate, or volatile anesthetic. "
                               "Check for recent surgery or procedural sedation.",
            "BARBITURATE_SITE": "Probable barbiturate exposure. "
                                "High risk of respiratory depression."
        }
        return interpretations.get(site, "Unknown binding site pattern.")
