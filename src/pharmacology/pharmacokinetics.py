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
        self.k_brain_in = params.bbb_permeability * 0.1  # Plasma → Brain (1/h)
        self.k_brain_out = 0.05  # Brain → Plasma (1/h)

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

        Reference: Schnider TW et al. (1998) Anesthesiology 88:1170
        """
        return PKParameters(
            bioavailability=1.0,  # IV only
            absorption_rate_constant=0.0,  # N/A for IV
            volume_of_distribution_L_kg=4.0,  # Large Vd (lipophilic)
            protein_binding_fraction=0.97,  # Highly protein-bound
            clearance_L_h_kg=1.8,  # High hepatic clearance
            half_life_hours=0.5,  # 30 minutes (rapid)
            bbb_permeability=0.9,  # Highly lipophilic → good BBB penetration
            brain_partition_coefficient=2.0,  # Concentrates in brain
        )

    @staticmethod
    def get_ketamine_pk() -> PKParameters:
        """
        Ketamine PK parameters.

        Reference: Clements JA, Nimmo WS (1981) Br J Anaesth 53:27
        """
        return PKParameters(
            bioavailability=0.93,  # IV ≈ 100%, IM ≈ 93%
            absorption_rate_constant=1.5,  # Rapid IM absorption
            volume_of_distribution_L_kg=3.0,
            protein_binding_fraction=0.53,
            clearance_L_h_kg=0.9,
            half_life_hours=2.5,
            bbb_permeability=0.85,  # Good BBB penetration
            brain_partition_coefficient=1.5,
        )

    @staticmethod
    def get_levodopa_pk() -> PKParameters:
        """
        Levodopa PK parameters.

        Reference: Contin M, Martinelli P (2010) Clin Pharmacokinet 49:141
        """
        return PKParameters(
            bioavailability=0.30,  # Low (extensive first-pass metabolism)
            absorption_rate_constant=2.0,  # Rapid absorption
            volume_of_distribution_L_kg=0.8,  # Small Vd (hydrophilic)
            protein_binding_fraction=0.10,  # Low protein binding
            clearance_L_h_kg=0.5,
            half_life_hours=1.5,  # Short half-life
            bbb_permeability=0.6,  # LAT1 transporter-mediated
            brain_partition_coefficient=0.8,
        )

    @staticmethod
    def get_fluoxetine_pk() -> PKParameters:
        """
        Fluoxetine PK parameters.

        Reference: Hiemke C, Hartter S (2000) Pharmacol Ther 85:11
        """
        return PKParameters(
            bioavailability=0.72,  # Good oral bioavailability
            absorption_rate_constant=0.5,  # Slow absorption
            volume_of_distribution_L_kg=20.0,  # Very large Vd (lipophilic)
            protein_binding_fraction=0.95,  # Highly protein-bound
            clearance_L_h_kg=0.02,  # Low clearance (long half-life)
            half_life_hours=96.0,  # 4 days (very long)
            bbb_permeability=0.7,
            brain_partition_coefficient=1.2,
        )

    @staticmethod
    def get_diazepam_pk() -> PKParameters:
        """
        Diazepam PK parameters.

        Reference: Olkkola KT, Ahonen J (2008) Clin Pharmacokinet 47:469
        """
        return PKParameters(
            bioavailability=1.0,  # Complete oral absorption
            absorption_rate_constant=1.0,
            volume_of_distribution_L_kg=1.1,
            protein_binding_fraction=0.98,  # Very high protein binding
            clearance_L_h_kg=0.016,  # Low clearance
            half_life_hours=43.0,  # Long half-life (20-100h range)
            bbb_permeability=0.85,
            brain_partition_coefficient=1.5,
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
