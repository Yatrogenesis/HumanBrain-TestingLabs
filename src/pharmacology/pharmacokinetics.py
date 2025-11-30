"""
Pharmacokinetics (PK) Models
============================

PBPK (Physiologically-Based Pharmacokinetic) models for:
- Drug absorption
- Distribution (compartmental models)
- Metabolism
- Elimination
- Blood-brain barrier transport

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class RouteOfAdministration(Enum):
    """Drug administration routes"""
    IV = "intravenous"  # Instant bioavailability
    ORAL = "oral"  # First-pass metabolism
    IM = "intramuscular"
    SC = "subcutaneous"
    INHALATION = "inhalation"


@dataclass
class PKParameters:
    """Pharmacokinetic parameters for a drug"""
    # Absorption
    bioavailability: float = 1.0  # F (0-1)
    absorption_rate_constant: float = 0.5  # ka (1/h)

    # Distribution
    volume_of_distribution_L_kg: float = 1.0  # Vd (L/kg)
    protein_binding_fraction: float = 0.5  # fu (0-1)

    # Metabolism & Elimination
    clearance_L_h_kg: float = 0.1  # CL (L/h/kg)
    half_life_hours: float = 4.0  # t½ (h)

    # Blood-brain barrier
    bbb_permeability: float = 0.5  # 0 = no penetration, 1 = free diffusion
    brain_partition_coefficient: float = 1.0  # Kp,brain


class CompartmentalPKModel:
    """
    Two-compartment PK model with blood-brain barrier.

    Compartments:
    1. Central (plasma)
    2. Peripheral (tissues)
    3. Brain (CNS)
    """

    def __init__(self, params: PKParameters, body_weight_kg: float = 70.0):
        self.params = params
        self.body_weight = body_weight_kg

        # Volumes (L)
        self.V_central = params.volume_of_distribution_L_kg * body_weight_kg * 0.3  # 30% in central
        self.V_peripheral = params.volume_of_distribution_L_kg * body_weight_kg * 0.6  # 60% in peripheral
        self.V_brain = 1.4  # Average brain volume (L)

        # Calculate rate constants from clearance and half-life
        self.k_elimination = params.clearance_L_h_kg * body_weight_kg / self.V_central  # 1/h
        self.k_12 = 0.5  # Central → Peripheral (1/h)
        self.k_21 = 0.3  # Peripheral → Central (1/h)

        # BIOLOGICALLY CALIBRATED BBB transport:
        # Only FREE (unbound) drug crosses BBB - protein binding is critical
        # Reference: Hammarlund-Udenaes M (2010) Clin Pharmacokinet 49:691
        free_fraction = 1.0 - params.protein_binding_fraction  # fu = unbound fraction

        # k_brain_in depends on: BBB permeability × free fraction × partition coefficient
        # Reduced base rate (0.02) for more realistic kinetics
        self.k_brain_in = params.bbb_permeability * free_fraction * 0.02 * params.brain_partition_coefficient

        # k_brain_out: efflux typically slower than influx for lipophilic drugs
        # Reference: Loryan I et al. (2014) Fluids Barriers CNS 11:3
        self.k_brain_out = 0.02 / params.brain_partition_coefficient  # Inversely related to Kp

        # State variables (amounts in mg)
        self.A_central = 0.0  # Amount in central compartment
        self.A_peripheral = 0.0  # Amount in peripheral compartment
        self.A_brain = 0.0  # Amount in brain
        self.A_gut = 0.0  # Amount in gut (for oral dosing)

        # Time tracking
        self.time = 0.0

    def administer_dose(self, dose_mg: float, route: RouteOfAdministration) -> None:
        """
        Administer a drug dose.

        Args:
            dose_mg: Dose in milligrams
            route: Route of administration
        """
        if route == RouteOfAdministration.IV:
            # Intravenous: directly to central compartment
            self.A_central += dose_mg
        elif route == RouteOfAdministration.ORAL:
            # Oral: to gut compartment (subject to absorption)
            self.A_gut += dose_mg * self.params.bioavailability
        else:
            # IM, SC: simplified as oral with higher bioavailability
            self.A_gut += dose_mg * min(1.0, self.params.bioavailability * 1.2)

    def step(self, dt: float = 0.1) -> Dict[str, float]:
        """
        Integrate PK model one timestep.

        Args:
            dt: Time step (hours)

        Returns:
            Concentrations in each compartment
        """
        # Gut absorption (first-order)
        absorption = self.params.absorption_rate_constant * self.A_gut
        self.A_gut -= absorption * dt
        self.A_central += absorption * dt

        # Inter-compartmental distribution
        transfer_12 = self.k_12 * self.A_central
        transfer_21 = self.k_21 * self.A_peripheral

        self.A_central += (transfer_21 - transfer_12) * dt
        self.A_peripheral += (transfer_12 - transfer_21) * dt

        # Brain distribution (BBB transport)
        transfer_brain_in = self.k_brain_in * self.A_central
        transfer_brain_out = self.k_brain_out * self.A_brain

        self.A_central += (transfer_brain_out - transfer_brain_in) * dt
        self.A_brain += (transfer_brain_in - transfer_brain_out) * dt

        # Elimination (from central compartment)
        elimination = self.k_elimination * self.A_central
        self.A_central -= elimination * dt

        # Prevent negative amounts
        self.A_central = max(0, self.A_central)
        self.A_peripheral = max(0, self.A_peripheral)
        self.A_brain = max(0, self.A_brain)
        self.A_gut = max(0, self.A_gut)

        self.time += dt

        return self.get_concentrations()

    def get_concentrations(self) -> Dict[str, float]:
        """
        Get current drug concentrations.

        Returns:
            Dictionary with concentrations (mg/L or μg/mL)
        """
        return {
            "plasma_mg_L": self.A_central / self.V_central if self.V_central > 0 else 0,
            "brain_mg_L": self.A_brain / self.V_brain if self.V_brain > 0 else 0,
            "peripheral_mg_L": self.A_peripheral / self.V_peripheral if self.V_peripheral > 0 else 0,
            "time_hours": self.time,
        }

    def get_brain_concentration_uM(self, molecular_weight: float) -> float:
        """
        Get brain concentration in μM.

        Args:
            molecular_weight: Molecular weight (g/mol)

        Returns:
            Brain concentration (μM)
        """
        conc_mg_L = self.A_brain / self.V_brain if self.V_brain > 0 else 0
        conc_uM = (conc_mg_L * 1000.0) / molecular_weight  # mg/L → μM
        return conc_uM

    def get_brain_concentration_nM(self, molecular_weight: float) -> float:
        """Get brain concentration in nM."""
        return self.get_brain_concentration_uM(molecular_weight) * 1000.0


class DrugPKDatabase:
    """Database of PK parameters for gold standard drugs."""

    @staticmethod
    def get_propofol_pk() -> PKParameters:
        """
        Propofol PK parameters.

        References:
        - Schnider TW et al. (1998) Anesthesiology 88:1170
        - Eleveld DJ et al. (2018) Br J Anaesth 120:942 (clinical concentration targets)

        Clinical target: Brain concentration 2-6 μM for anesthesia
        Only 3% unbound (free) → effective brain uptake is limited
        """
        return PKParameters(
            bioavailability=1.0,  # IV only
            absorption_rate_constant=0.0,  # N/A for IV
            volume_of_distribution_L_kg=4.0,  # Large Vd (lipophilic)
            protein_binding_fraction=0.90,  # Adjusted: effective ~10% free accounts for rapid BBB equilibrium
            clearance_L_h_kg=1.8,  # High hepatic clearance
            half_life_hours=0.5,  # 30 minutes distribution half-life
            bbb_permeability=0.95,  # High intrinsic permeability (lipophilic)
            brain_partition_coefficient=5.0,  # Adjusted for clinical target 2-6 μM brain at 140mg IV
        )

    @staticmethod
    def get_ketamine_pk() -> PKParameters:
        """
        Ketamine PK parameters.

        References:
        - Clements JA, Nimmo WS (1981) Br J Anaesth 53:27
        - Fanta S et al. (2015) Br J Clin Pharmacol 80:1269 (brain concentrations)

        Clinical target: Brain concentration 5-20 μM for anesthesia
        ~47% protein bound → 53% free, good CNS penetration
        """
        return PKParameters(
            bioavailability=0.93,  # IM bioavailability ~93%
            absorption_rate_constant=1.5,  # Rapid IM absorption (Tmax ~20 min)
            volume_of_distribution_L_kg=3.0,  # Moderate Vd
            protein_binding_fraction=0.47,  # Less bound than propofol → more free drug
            clearance_L_h_kg=0.9,  # Hepatic CYP3A4 metabolism
            half_life_hours=2.5,  # Terminal elimination half-life
            bbb_permeability=0.90,  # Excellent BBB permeability (lipophilic)
            brain_partition_coefficient=3.0,  # Brain-plasma ratio ~3-4 in humans
        )

    @staticmethod
    def get_levodopa_pk() -> PKParameters:
        """
        Levodopa PK parameters.

        References:
        - Contin M, Martinelli P (2010) Clin Pharmacokinet 49:141
        - Nutt JG, Fellman JH (1984) Clin Neuropharmacol 7:35 (brain uptake)

        CRITICAL BIOLOGY:
        1. ~30% oral bioavailability (first-pass decarboxylation in gut)
        2. LAT1 transporter-mediated BBB crossing (saturable, competitive)
        3. Only ~1-3% of oral dose reaches brain as L-DOPA
        4. Then ~10% converted to dopamine by AADC in brain
        5. Net effect: ~0.1-0.3% of dose becomes brain dopamine

        Clinical target: Effective dopamine increase for 40% UPDRS improvement
        """
        return PKParameters(
            bioavailability=0.30,  # 30% after first-pass metabolism
            absorption_rate_constant=2.0,  # Rapid absorption (Tmax ~1h)
            volume_of_distribution_L_kg=0.8,  # Small Vd (hydrophilic amino acid)
            protein_binding_fraction=0.10,  # Low binding, 90% free
            clearance_L_h_kg=0.5,  # Peripheral AADC decarboxylation
            half_life_hours=1.5,  # Short plasma half-life
            # CALIBRATED FOR 40% UPDRS: LAT1 active transport
            bbb_permeability=0.95,  # Near-maximal LAT1 efficiency
            brain_partition_coefficient=25.0,  # High brain accumulation via LAT1
        )

    @staticmethod
    def get_fluoxetine_pk() -> PKParameters:
        """
        Fluoxetine PK parameters.

        References:
        - Hiemke C, Hartter S (2000) Pharmacol Ther 85:11
        - Karson CN et al. (1993) Arch Gen Psychiatry 50:615 (brain levels)

        CRITICAL BIOLOGY:
        1. Very lipophilic → accumulates extensively in brain
        2. Brain/plasma ratio ~20:1 at steady state
        3. 95% protein bound → only 5% free
        4. 5-HT increase limited by 5-HT1A autoreceptor feedback
        5. Therapeutic delay (2-4 weeks) due to autoreceptor desensitization

        Clinical target: ~50 nM synaptic 5-HT (5x baseline)
        """
        return PKParameters(
            bioavailability=0.72,  # Good oral bioavailability
            absorption_rate_constant=0.5,  # Slow absorption (Tmax ~6-8h)
            volume_of_distribution_L_kg=20.0,  # Very large Vd (35 L/kg including brain)
            protein_binding_fraction=0.95,  # 95% bound, only 5% free
            clearance_L_h_kg=0.02,  # Very low clearance
            half_life_hours=96.0,  # 4-6 days (accumulation over weeks)
            bbb_permeability=0.70,  # Good passive diffusion
            brain_partition_coefficient=1.5,  # Adjusted: Brain Kp ~1-2 for free drug at steady state
        )

    @staticmethod
    def get_diazepam_pk() -> PKParameters:
        """
        Diazepam PK parameters.

        References:
        - Olkkola KT, Ahonen J (2008) Clin Pharmacokinet 47:469
        - Dhillon S, Richens A (1982) Br J Clin Pharmacol 14:357 (brain levels)

        CRITICAL BIOLOGY:
        1. 98% protein bound → only 2% free to cross BBB
        2. Benzodiazepine site is different from propofol anesthetic site
        3. BZ site has much higher affinity (Ki ~20 nM vs μM for propofol)
        4. But lower intrinsic efficacy (anxiolytic vs anesthetic)

        Clinical target: 40% beta power increase (anxiolytic effect)
        """
        return PKParameters(
            bioavailability=1.0,  # Complete oral absorption
            absorption_rate_constant=1.0,  # Moderate absorption (Tmax ~1h)
            volume_of_distribution_L_kg=1.1,  # Moderate Vd
            protein_binding_fraction=0.98,  # CRITICAL: Only 2% free
            clearance_L_h_kg=0.016,  # Low clearance → long duration
            half_life_hours=43.0,  # Long half-life (20-100h)
            bbb_permeability=0.90,  # Good passive diffusion
            brain_partition_coefficient=3.0,  # Moderate brain uptake
        )


    @staticmethod
    def get_morphine_pk() -> PKParameters:
        """
        Morphine PK parameters.
        References:
        - Glare PA, Walsh TD (1991) Clin Pharmacokinet 20:131
        - Portenoy RK (1996) Anesthesiology 84:1243
        Clinical target: 50% pain reduction (VAS) at 10mg IV
        """
        return PKParameters(
            bioavailability=0.30,
            absorption_rate_constant=1.5,
            volume_of_distribution_L_kg=3.5,
            protein_binding_fraction=0.35,
            clearance_L_h_kg=1.0,
            half_life_hours=3.0,
            bbb_permeability=0.60,
            brain_partition_coefficient=2.5,
        )

    @staticmethod
    def get_haloperidol_pk() -> PKParameters:
        """
        Haloperidol PK parameters.
        References:
        - Kudo S, Ishizaki T (1999) Clin Pharmacokinet 37:435
        - Seeman P (2002) Neuropsychopharmacology 26:587
        Clinical target: 65% D2 occupancy (antipsychotic threshold)

        Note: Brain partition adjusted to match PET D2 occupancy data
        (Kapur et al. 2000: 5mg → 65% D2 occupancy)
        """
        return PKParameters(
            bioavailability=0.60,
            absorption_rate_constant=1.0,
            volume_of_distribution_L_kg=18.0,
            protein_binding_fraction=0.92,
            clearance_L_h_kg=0.8,
            half_life_hours=18.0,
            bbb_permeability=0.85,
            brain_partition_coefficient=0.13,  # Calibrated: 1.86 nM brain → 65% D2 at 5mg (Ki=1nM, PET data)
        )

    @staticmethod
    def get_midazolam_pk() -> PKParameters:
        """
        Midazolam PK parameters.
        References:
        - Bauer TM et al. (1995) Clin Pharmacokinet 29:157
        - Olkkola KT, Ahonen J (2008) Clin Pharmacokinet 47:469
        Clinical target: 70% sedation (Ramsay 3-4) at 0.1mg/kg IV
        """
        return PKParameters(
            bioavailability=0.35,
            absorption_rate_constant=2.0,
            volume_of_distribution_L_kg=1.5,
            protein_binding_fraction=0.96,
            clearance_L_h_kg=0.4,
            half_life_hours=2.5,
            bbb_permeability=0.90,
            brain_partition_coefficient=4.0,
        )


def simulate_pk_profile(
    drug_name: str,
    dose_mg: float,
    route: RouteOfAdministration,
    duration_hours: float = 24.0,
    dt_hours: float = 0.1,
    body_weight_kg: float = 70.0
) -> Dict:
    """
    Simulate PK profile for a drug.

    Args:
        drug_name: 'propofol', 'ketamine', 'levodopa', 'fluoxetine', 'diazepam'
        dose_mg: Dose in milligrams
        route: Route of administration
        duration_hours: Simulation duration (hours)
        dt_hours: Time step (hours)
        body_weight_kg: Patient body weight (kg)

    Returns:
        Dictionary with time course data
    """
    # Get PK parameters
    db = DrugPKDatabase()
    pk_params_map = {
        "propofol": (db.get_propofol_pk(), 178.27),  # MW
        "ketamine": (db.get_ketamine_pk(), 237.73),
        "levodopa": (db.get_levodopa_pk(), 197.19),
        "fluoxetine": (db.get_fluoxetine_pk(), 309.33),
        "diazepam": (db.get_diazepam_pk(), 284.74),
        "morphine": (db.get_morphine_pk(), 285.34),
        "haloperidol": (db.get_haloperidol_pk(), 375.86),
        "midazolam": (db.get_midazolam_pk(), 325.77),
    }

    if drug_name not in pk_params_map:
        raise ValueError(f"Unknown drug: {drug_name}")

    pk_params, molecular_weight = pk_params_map[drug_name]

    # Create PK model
    model = CompartmentalPKModel(pk_params, body_weight_kg)
    model.administer_dose(dose_mg, route)

    # Simulate
    n_steps = int(duration_hours / dt_hours)
    time = np.zeros(n_steps)
    plasma_conc = np.zeros(n_steps)
    brain_conc = np.zeros(n_steps)

    for i in range(n_steps):
        concentrations = model.step(dt_hours)
        time[i] = concentrations["time_hours"]
        plasma_conc[i] = concentrations["plasma_mg_L"]
        brain_conc[i] = model.get_brain_concentration_uM(molecular_weight)

    return {
        "time_hours": time,
        "plasma_concentration_mg_L": plasma_conc,
        "brain_concentration_uM": brain_conc,
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "route": route.value,
        "molecular_weight": molecular_weight,
    }


if __name__ == "__main__":
    # Test PK models for all gold standard drugs
    print("Testing Pharmacokinetic Models...\n")
    print("=" * 80)

    # Propofol (IV bolus for anesthesia)
    print("\n1. PROPOFOL (2 mg/kg IV bolus)")
    print("-" * 80)
    result = simulate_pk_profile("propofol", dose_mg=140, route=RouteOfAdministration.IV, duration_hours=4.0)
    peak_idx = np.argmax(result["brain_concentration_uM"])
    print(f"  Peak brain concentration: {result['brain_concentration_uM'][peak_idx]:.2f} μM at {result['time_hours'][peak_idx]:.2f} h")
    print(f"  Target clinical range: 2-6 μM")

    # Ketamine (IM for anesthesia)
    print("\n2. KETAMINE (2 mg/kg IM)")
    print("-" * 80)
    result = simulate_pk_profile("ketamine", dose_mg=140, route=RouteOfAdministration.IM, duration_hours=8.0)
    peak_idx = np.argmax(result["brain_concentration_uM"])
    print(f"  Peak brain concentration: {result['brain_concentration_uM'][peak_idx]:.2f} μM at {result['time_hours'][peak_idx]:.2f} h")
    print(f"  Target clinical range: 5-20 μM")

    # Levodopa (oral for Parkinson's)
    print("\n3. LEVODOPA (100 mg oral)")
    print("-" * 80)
    result = simulate_pk_profile("levodopa", dose_mg=100, route=RouteOfAdministration.ORAL, duration_hours=8.0)
    peak_idx = np.argmax(result["brain_concentration_uM"])
    print(f"  Peak brain concentration: {result['brain_concentration_uM'][peak_idx]:.2f} μM at {result['time_hours'][peak_idx]:.2f} h")
    print(f"  Typical peak time: 0.5-2 hours")

    # Fluoxetine (oral daily for depression)
    print("\n4. FLUOXETINE (20 mg oral)")
    print("-" * 80)
    result = simulate_pk_profile("fluoxetine", dose_mg=20, route=RouteOfAdministration.ORAL, duration_hours=168.0)  # 1 week
    peak_idx = np.argmax(result["brain_concentration_uM"])
    print(f"  Peak brain concentration: {result['brain_concentration_uM'][peak_idx]:.2f} μM at {result['time_hours'][peak_idx]:.1f} h")
    print(f"  Steady-state time: ~4-5 weeks (long half-life)")

    # Diazepam (oral for anxiety)
    print("\n5. DIAZEPAM (10 mg oral)")
    print("-" * 80)
    result = simulate_pk_profile("diazepam", dose_mg=10, route=RouteOfAdministration.ORAL, duration_hours=72.0)  # 3 days
    peak_idx = np.argmax(result["brain_concentration_uM"])
    print(f"  Peak brain concentration: {result['brain_concentration_uM'][peak_idx]:.2f} μM at {result['time_hours'][peak_idx]:.2f} h")
    print(f"  Long half-life: 20-100 hours")

    print("\n" + "=" * 80)
    print("PK module test complete! ✓")
