"""
Neural Network Builder for Human Brain Simulation
=================================================

Builds multi-region brain networks with pharmacologically-modulated synapses.

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    # When running from package
    from simulation.neuron_models import HodgkinHuxleyNeuron, IzhikevichNeuron, LIFNeuron
    from simulation.synapse_models import GABAaReceptor, NMDAReceptor, DopamineD2Receptor, SerotoninTransporter
except ImportError:
    # When running directly
    from neuron_models import HodgkinHuxleyNeuron, IzhikevichNeuron, LIFNeuron
    from synapse_models import GABAaReceptor, NMDAReceptor, DopamineD2Receptor, SerotoninTransporter


# ============================================================================
# GENERALIZED BRAIN MECHANISTIC MODEL
# ============================================================================
# This model implements the biophysical cascade from drug binding to clinical
# effect using first-principles pharmacology and neuroscience.
#
# CASCADE: Drug → Receptor → Ion Channel → ΔV_membrane → ΔFiring → ΔNetwork → Effect
#
# Key References:
#   - Black JW, Leff P (1983) Proc R Soc Lond B 220:141 (Operational model)
#   - Rang HP (2006) Brit J Pharmacol 147:S72 (Receptor reserve)
#   - Franks NP (2008) Brit J Pharmacol 153:S72 (Anesthetic mechanisms)
#   - Jahr CE, Stevens CF (1990) J Neurosci 10:3178 (NMDA receptor)
# ============================================================================

@dataclass
class IonChannelParameters:
    """Biophysical parameters for ion channels (from literature)."""
    # GABA_A receptor (Cl- channel) - Olsen & Sieghart 2008
    g_gaba_pS: float = 30.0          # Single channel conductance (pS)
    E_cl_mV: float = -75.0           # Chloride reversal potential (mV)
    n_gaba_per_synapse: int = 50     # Receptors per synapse
    tau_gaba_ms: float = 30.0        # Decay time constant (ms)

    # NMDA receptor (Ca2+/Na+/K+) - Jahr & Stevens 1990
    g_nmda_pS: float = 50.0          # Single channel conductance (pS)
    E_nmda_mV: float = 0.0           # Reversal potential (mV)
    n_nmda_per_synapse: int = 20     # Receptors per synapse
    tau_nmda_ms: float = 100.0       # Decay time constant (ms)
    mg_block_mM: float = 1.0         # Extracellular Mg2+ (mM)

    # AMPA receptor (Na+/K+) - Traynelis et al. 2010
    g_ampa_pS: float = 20.0          # Single channel conductance (pS)
    E_ampa_mV: float = 0.0           # Reversal potential (mV)


@dataclass
class ReceptorReserveParameters:
    """
    Receptor reserve (spare receptors) parameters.

    Based on Black & Leff (1983) operational model:
    Effect = E_max × [Occupancy × τ] / [1 + Occupancy × (τ - 1)]

    Where τ (tau) is the receptor reserve factor:
    - τ >> 1: large reserve, full effect before full occupancy
    - τ ~ 1: no reserve, linear occupancy-effect relationship
    - τ < 1: receptor deficit, never reaches full effect

    Values from literature for different systems.
    """
    # GABAergic system - high reserve (Barnard et al. 1998)
    tau_gaba_cortex: float = 5.0      # ~80% reserve in cortex
    tau_gaba_brainstem: float = 3.0   # ~67% reserve in brainstem

    # Glutamatergic system - moderate reserve (Bhattacharyya 2016)
    tau_nmda: float = 2.5             # ~60% reserve
    tau_ampa: float = 2.0             # ~50% reserve

    # Dopaminergic system - variable by region (Richfield et al. 1989)
    tau_d1_striatum: float = 8.0      # ~87% reserve (D1 has high reserve)
    tau_d2_striatum: float = 3.0      # ~67% reserve

    # Serotonergic system - high reserve (Millan et al. 2008)
    tau_5ht1a: float = 10.0           # ~90% reserve
    tau_sert: float = 4.0             # ~75% reserve for transporter


@dataclass
class NetworkIntegrationParameters:
    """
    Parameters for how individual neuron effects integrate at network level.

    Based on:
    - Cortical E/I balance (Isaacson & Scanziani 2011)
    - Neural population dynamics (Wilson & Cowan 1972)
    """
    # Excitatory/Inhibitory balance
    ei_ratio_cortex: float = 0.80     # E/I ratio (typical 80:20)
    ei_ratio_striatum: float = 0.95   # Mostly inhibitory projection neurons

    # Network connectivity (fraction of neurons affected)
    gaba_network_fraction: float = 0.20   # 20% are GABAergic interneurons
    nmda_network_fraction: float = 0.80   # 80% receive glutamatergic input
    dopamine_network_fraction: float = 0.15  # 15% in striatum receive DA

    # Oscillation coupling (for EEG effects)
    beta_gaba_coupling: float = 0.40   # GABA → beta (13-30 Hz)
    gamma_nmda_coupling: float = 0.50  # NMDA blockade → gamma (30-80 Hz)


# Global instances of biophysical parameters
ION_CHANNELS = IonChannelParameters()
RECEPTOR_RESERVE = ReceptorReserveParameters()
NETWORK_PARAMS = NetworkIntegrationParameters()


def operational_model_effect(occupancy: float, tau: float, e_max: float = 1.0) -> float:
    """
    Black & Leff (1983) operational model for receptor-effect relationship.

    This accounts for receptor reserve (spare receptors).

    Args:
        occupancy: Fraction of receptors occupied (0-1)
        tau: Receptor reserve parameter (higher = more reserve)
        e_max: Maximum possible effect

    Returns:
        Effect as fraction of E_max (0-1)

    Reference:
        Black JW, Leff P (1983) Proc R Soc Lond B 220:141
    """
    if tau <= 0 or occupancy <= 0:
        return 0.0

    # Operational model equation
    effect = e_max * (occupancy * tau) / (1.0 + occupancy * (tau - 1.0))
    return min(e_max, max(0.0, effect))


def calculate_conductance_change(
    occupancy: float,
    n_receptors: int,
    g_single_pS: float,
    efficacy: float = 1.0
) -> float:
    """
    Calculate change in membrane conductance from receptor activation.

    Args:
        occupancy: Fraction of receptors occupied
        n_receptors: Number of receptors per synapse
        g_single_pS: Single channel conductance (pS)
        efficacy: Drug efficacy (0-1)

    Returns:
        Total conductance change (nS)
    """
    # Total conductance = N × g × P_open × efficacy
    # P_open approximated by occupancy for ligand-gated channels
    g_total_pS = n_receptors * g_single_pS * occupancy * efficacy
    return g_total_pS / 1000.0  # Convert to nS


def calculate_firing_rate_change(
    delta_g_inhibitory_nS: float,
    delta_g_excitatory_nS: float,
    baseline_rate_Hz: float = 10.0
) -> float:
    """
    Calculate change in neuronal firing rate from conductance changes.

    Based on leaky integrate-and-fire approximation.

    Args:
        delta_g_inhibitory_nS: Change in inhibitory conductance
        delta_g_excitatory_nS: Change in excitatory conductance
        baseline_rate_Hz: Baseline firing rate

    Returns:
        New firing rate (Hz)
    """
    # Simplified model: firing rate ∝ (E_drive - V_threshold) / τ
    # Inhibitory conductance increases threshold, excitatory decreases it

    # Membrane time constant (typical cortical neuron)
    tau_membrane_ms = 20.0
    g_leak_nS = 10.0  # Leak conductance

    # Effective change in firing rate
    inhibitory_factor = delta_g_inhibitory_nS / (g_leak_nS + delta_g_inhibitory_nS + 0.001)
    excitatory_factor = delta_g_excitatory_nS / (g_leak_nS + delta_g_excitatory_nS + 0.001)

    # New firing rate (cannot go below 0)
    new_rate = baseline_rate_Hz * (1.0 - inhibitory_factor + excitatory_factor)
    return max(0.0, new_rate)


class BrainRegion(Enum):
    """Major brain regions for pharmacological simulation"""
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    MOTOR_CORTEX = "motor_cortex"
    STRIATUM = "striatum"  # Dopamine-rich (Parkinson's)
    THALAMUS = "thalamus"
    BRAINSTEM = "brainstem"
    RAPHE_NUCLEI = "raphe_nuclei"  # Serotonin source


class NeuronType(Enum):
    """Neuron types with different pharmacological sensitivity"""
    EXCITATORY_PYRAMIDAL = "excitatory_pyramidal"  # Glutamatergic
    INHIBITORY_INTERNEURON = "inhibitory_interneuron"  # GABAergic
    DOPAMINERGIC = "dopaminergic"  # Substantia nigra, VTA
    SEROTONERGIC = "serotonergic"  # Raphe nuclei


@dataclass
class NetworkParameters:
    """Parameters for brain network construction"""
    # Total neurons (scaled for hardware)
    n_neurons_total: int = 1_000_000  # 1M for M1, 100M for RTX3050

    # Region distribution (percentages)
    region_distribution: Dict[BrainRegion, float] = None

    # Connection probability
    p_local_connection: float = 0.1  # Within-region
    p_inter_region: float = 0.01  # Between-region

    # Synaptic weights (nS)
    w_excitatory: float = 0.5
    w_inhibitory: float = 1.0

    def __post_init__(self):
        if self.region_distribution is None:
            self.region_distribution = {
                BrainRegion.PREFRONTAL_CORTEX: 0.30,
                BrainRegion.MOTOR_CORTEX: 0.25,
                BrainRegion.STRIATUM: 0.20,
                BrainRegion.THALAMUS: 0.15,
                BrainRegion.BRAINSTEM: 0.08,
                BrainRegion.RAPHE_NUCLEI: 0.02,
            }


class BrainNetwork:
    """
    Multi-region brain network with pharmacologically-modulated synapses.
    """

    def __init__(self, params: NetworkParameters = None):
        self.params = params or NetworkParameters()

        # Network structure
        self.neurons: List[object] = []  # Neuron objects
        self.neuron_regions: List[BrainRegion] = []  # Region assignments
        self.neuron_types: List[NeuronType] = []  # Type assignments

        # Synaptic connections
        self.connections: List[Tuple[int, int, float, str]] = []  # (pre, post, weight, receptor_type)

        # Receptor populations
        self.gaba_receptors: Dict[int, GABAaReceptor] = {}  # Neuron ID -> receptor
        self.nmda_receptors: Dict[int, NMDAReceptor] = {}
        self.d2_receptors: Dict[int, DopamineD2Receptor] = {}
        self.sert_transporters: Dict[int, SerotoninTransporter] = {}

        # State variables
        self.neurotransmitter_concentrations: Dict[str, float] = {
            "gaba": 1.0,  # μM
            "glutamate": 2.0,  # μM
            "dopamine": 10.0,  # nM (low baseline in Parkinson's)
            "serotonin": 10.0,  # nM
        }

    def build_network(self, neuron_model: str = "izhikevich") -> None:
        """
        Construct brain network with specified neuron model.

        Args:
            neuron_model: 'hodgkin_huxley', 'izhikevich', or 'lif'
        """
        print(f"Building brain network with {self.params.n_neurons_total:,} neurons ({neuron_model})...")

        # 1. Create neurons by region
        for region, fraction in self.params.region_distribution.items():
            n_region = int(self.params.n_neurons_total * fraction)
            self._create_region(region, n_region, neuron_model)

        # 2. Create synaptic connections
        self._connect_neurons()

        # 3. Initialize receptors
        self._initialize_receptors()

        print(f"Network built: {len(self.neurons):,} neurons, {len(self.connections):,} synapses ✓")

    def _create_region(self, region: BrainRegion, n_neurons: int, model: str) -> None:
        """Create neurons for a specific brain region."""
        for _ in range(n_neurons):
            # Select neuron type based on region
            if region == BrainRegion.STRIATUM:
                neuron_type = NeuronType.DOPAMINERGIC if np.random.rand() < 0.1 else NeuronType.EXCITATORY_PYRAMIDAL
            elif region == BrainRegion.RAPHE_NUCLEI:
                neuron_type = NeuronType.SEROTONERGIC
            else:
                # Cortical regions: 80% excitatory, 20% inhibitory
                neuron_type = (
                    NeuronType.EXCITATORY_PYRAMIDAL
                    if np.random.rand() < 0.8
                    else NeuronType.INHIBITORY_INTERNEURON
                )

            # Create neuron model
            if model == "hodgkin_huxley":
                neuron = HodgkinHuxleyNeuron()
            elif model == "izhikevich":
                if neuron_type == NeuronType.INHIBITORY_INTERNEURON:
                    neuron = IzhikevichNeuron("fast_spiking")
                else:
                    neuron = IzhikevichNeuron("regular_spiking")
            else:  # lif
                neuron = LIFNeuron()

            self.neurons.append(neuron)
            self.neuron_regions.append(region)
            self.neuron_types.append(neuron_type)

    def _connect_neurons(self) -> None:
        """Create synaptic connections between neurons."""
        n = len(self.neurons)

        for i in range(n):
            # Local connections (within region)
            region_i = self.neuron_regions[i]
            for j in range(n):
                if i == j:
                    continue

                region_j = self.neuron_regions[j]

                # Connection probability
                if region_i == region_j:
                    p_connect = self.params.p_local_connection
                else:
                    p_connect = self.params.p_inter_region

                if np.random.rand() < p_connect:
                    # Determine synapse type
                    neuron_type_i = self.neuron_types[i]

                    if neuron_type_i == NeuronType.INHIBITORY_INTERNEURON:
                        receptor_type = "gaba_a"
                        weight = -self.params.w_inhibitory  # Negative = inhibitory
                    elif neuron_type_i == NeuronType.DOPAMINERGIC:
                        receptor_type = "dopamine_d2"
                        weight = self.params.w_excitatory * 0.5  # Modulatory
                    elif neuron_type_i == NeuronType.SEROTONERGIC:
                        receptor_type = "serotonin_5ht"
                        weight = self.params.w_excitatory * 0.3  # Modulatory
                    else:  # Excitatory
                        receptor_type = "nmda"
                        weight = self.params.w_excitatory

                    self.connections.append((i, j, weight, receptor_type))

    def _initialize_receptors(self) -> None:
        """Initialize receptor populations for each neuron."""
        for i, neuron_type in enumerate(self.neuron_types):
            # All neurons have GABA_A receptors (inhibition)
            self.gaba_receptors[i] = GABAaReceptor()

            # All neurons have NMDA receptors (excitation)
            self.nmda_receptors[i] = NMDAReceptor()

            # Dopamine D2 receptors (especially striatum)
            if self.neuron_regions[i] == BrainRegion.STRIATUM:
                self.d2_receptors[i] = DopamineD2Receptor()

            # Serotonin transporters (especially raphe nuclei)
            if self.neuron_regions[i] == BrainRegion.RAPHE_NUCLEI:
                self.sert_transporters[i] = SerotoninTransporter()

    def apply_drug(self, drug_name: str, concentration: float, unit: str = "uM") -> Dict:
        """
        Apply pharmacological agent to network.

        Args:
            drug_name: 'propofol', 'ketamine', 'levodopa', 'fluoxetine', 'diazepam'
            concentration: Drug concentration
            unit: Concentration unit ('uM', 'nM', 'mg/mL')

        Returns:
            Dictionary with drug effects
        """
        print(f"\nApplying {drug_name} ({concentration} {unit})...")

        effects = {
            "drug": drug_name,
            "concentration": concentration,
            "unit": unit,
            "receptor_targets": [],
            "network_effects": {},
        }

        if drug_name == "propofol":
            # GABA_A positive allosteric modulator
            for neuron_id, receptor in self.gaba_receptors.items():
                receptor.bind_drug(concentration, drug_type="propofol", efficacy=0.85)
            effects["receptor_targets"] = ["GABA_A"]
            effects["network_effects"]["eeg_suppression_pct"] = self._calculate_eeg_suppression()

        elif drug_name == "ketamine":
            # NMDA antagonist
            for neuron_id, receptor in self.nmda_receptors.items():
                receptor.bind_ketamine(concentration)
            effects["receptor_targets"] = ["NMDA"]
            effects["network_effects"]["gamma_power_increase"] = self._calculate_gamma_power()

        elif drug_name == "levodopa":
            # Dopamine precursor (convert to dopamine)
            # Typical conversion: 100 mg levodopa → ~5 μM brain dopamine
            dopamine_increase_nM = concentration * 50.0  # Empirical conversion
            self.neurotransmitter_concentrations["dopamine"] += dopamine_increase_nM

            for neuron_id, receptor in self.d2_receptors.items():
                receptor.bind_dopamine(self.neurotransmitter_concentrations["dopamine"])
            effects["receptor_targets"] = ["Dopamine D2"]
            effects["network_effects"]["motor_improvement_pct"] = self._calculate_motor_improvement()

        elif drug_name == "fluoxetine":
            # SERT inhibitor (SSRI)
            concentration_nM = concentration * 1000.0 if unit == "uM" else concentration
            for neuron_id, transporter in self.sert_transporters.items():
                transporter.bind_fluoxetine(concentration_nM)
            effects["receptor_targets"] = ["SERT"]
            effects["network_effects"]["serotonin_increase_nM"] = self._calculate_serotonin_increase()

        elif drug_name == "diazepam":
            # GABA_A benzodiazepine site
            for neuron_id, receptor in self.gaba_receptors.items():
                receptor.bind_drug(concentration, drug_type="diazepam", efficacy=0.70)
            effects["receptor_targets"] = ["GABA_A (BZ site)"]
            effects["network_effects"]["beta_power_increase_pct"] = self._calculate_beta_power()

        return effects

    def _calculate_eeg_suppression(self) -> float:
        """
        Calculate EEG suppression percentage (propofol effect).

        Mechanistic cascade:
        1. Drug binds GABA_A receptor (allosteric modulation)
        2. Enhanced Cl- conductance → hyperpolarization
        3. Reduced neuronal firing → reduced cortical activity
        4. EEG suppression (burst-suppression pattern)

        References:
            - Brown EN et al. (2011) NEJM 363:2638
            - Purdon PL et al. (2013) PNAS 110:E1142
        """
        # Step 1: Get average receptor occupancy and modulation
        modulations = [r.modulation_factor for r in self.gaba_receptors.values()]
        mean_modulation = np.mean(modulations)

        # Step 2: Calculate effective Cl- conductance change
        # Modulation factor represents how much GABA response is enhanced
        # Propofol both potentiates GABA and has direct agonist effect
        delta_g_cl = calculate_conductance_change(
            occupancy=mean_modulation - 1.0,  # Excess over baseline
            n_receptors=ION_CHANNELS.n_gaba_per_synapse,
            g_single_pS=ION_CHANNELS.g_gaba_pS,
            efficacy=0.85  # Propofol high efficacy
        )

        # Step 3: Apply receptor reserve (Black & Leff operational model)
        # For anesthetic depth, use cortical GABAergic reserve
        effective_modulation = operational_model_effect(
            occupancy=mean_modulation - 1.0,
            tau=RECEPTOR_RESERVE.tau_gaba_cortex,
            e_max=1.0
        )

        # Step 4: Calculate firing rate suppression
        baseline_firing = 10.0  # Hz (typical cortical neuron)
        new_firing = calculate_firing_rate_change(
            delta_g_inhibitory_nS=delta_g_cl * NETWORK_PARAMS.gaba_network_fraction,
            delta_g_excitatory_nS=0.0,
            baseline_rate_Hz=baseline_firing
        )

        # Step 5: EEG suppression relates to population firing reduction
        # Plus direct cortical depression from high GABA_A activation
        firing_suppression = (baseline_firing - new_firing) / baseline_firing
        eeg_suppression = 100.0 * (effective_modulation + firing_suppression) / 2.0

        # Cap at physiological maximum (burst-suppression = ~90%)
        return min(90.0, max(0.0, eeg_suppression))

    def _calculate_gamma_power(self) -> float:
        """
        Calculate gamma oscillation power increase (ketamine effect).

        Mechanistic cascade:
        1. Ketamine blocks NMDA receptors (open-channel blocker)
        2. Preferential block of NMDA on GABAergic interneurons
        3. Disinhibition of pyramidal neurons
        4. Increased gamma oscillations (30-80 Hz)

        This is the "disinhibition hypothesis" - ketamine preferentially
        affects fast-spiking interneurons which have higher NMDA/AMPA ratio.

        References:
            - Sleigh JW et al. (2014) Br J Anaesth 113:i61
            - Homayoun H, Bhagat M (2007) J Neurosci 27:11496
        """
        # Step 1: Get average NMDA blockade
        blockades = [r.blockade_fraction for r in self.nmda_receptors.values()]
        mean_blockade = np.mean(blockades)

        # Step 2: Apply receptor reserve for NMDA system
        effective_blockade = operational_model_effect(
            occupancy=mean_blockade,
            tau=RECEPTOR_RESERVE.tau_nmda,
            e_max=1.0
        )

        # Step 3: Calculate disinhibition
        # Interneurons have ~40% higher NMDA/AMPA ratio than pyramidal neurons
        # So they're preferentially affected by NMDA blockade
        interneuron_nmda_ratio = 1.4
        pyramidal_nmda_ratio = 1.0

        # Interneuron silencing leads to disinhibition
        interneuron_suppression = effective_blockade * interneuron_nmda_ratio
        disinhibition_factor = interneuron_suppression * NETWORK_PARAMS.gaba_network_fraction

        # Step 4: Gamma oscillation increase from disinhibition
        # Wilson-Cowan model: gamma emerges from E-I imbalance
        # Empirically: gamma increases ~2-3x at anesthetic doses
        baseline_gamma = 1.0
        gamma_coupling = NETWORK_PARAMS.gamma_nmda_coupling

        # Gamma power scales with disinhibition
        # The factor of 8 is derived from:
        #   - Interneurons are ~20% of cortical neurons
        #   - But they provide ~80% of inhibition (4x amplification)
        #   - Plus 1.4x higher NMDA/AMPA ratio → total ~5-6x effective disinhibition
        #   - Network resonance adds another ~30% (factor 1.3)
        # Combined: 4 × 1.4 × 1.3 ≈ 7.3, rounded to 8
        amplification_factor = 8.0

        gamma_increase = baseline_gamma * (1.0 + gamma_coupling * disinhibition_factor * amplification_factor)

        # Cap at physiological maximum (~4x increase observed in studies)
        return min(4.0, max(1.0, gamma_increase))

    def _calculate_motor_improvement(self) -> float:
        """
        Calculate motor function improvement (levodopa effect).

        Mechanistic cascade:
        1. Levodopa → Dopamine (via AADC in striatum)
        2. Dopamine binds D1/D2 receptors in striatum
        3. D1 activates direct pathway → facilitates movement
        4. D2 inhibits indirect pathway → reduces movement inhibition
        5. Net effect: improved motor function (UPDRS score)

        References:
            - Poewe W et al. (2017) Nat Rev Dis Primers 3:17013
            - DeLong MR, Wichmann T (2007) Lancet Neurol 6:352
        """
        if not self.d2_receptors:
            return 0.0

        # Step 1: Get dopamine concentrations
        baseline_da = 5.0  # nM (Parkinson's baseline - severely depleted)
        current_da = self.neurotransmitter_concentrations["dopamine"]

        # Step 2: Calculate D2 receptor activation
        # D2 Kd ~ 20 nM (from synapse_models.py)
        kd_d2 = 20.0  # nM

        baseline_d2_occupancy = baseline_da / (kd_d2 + baseline_da)
        current_d2_occupancy = current_da / (kd_d2 + current_da)

        # Step 3: Apply receptor reserve (D2 has moderate reserve)
        baseline_effect = operational_model_effect(
            occupancy=baseline_d2_occupancy,
            tau=RECEPTOR_RESERVE.tau_d2_striatum,
            e_max=1.0
        )
        current_effect = operational_model_effect(
            occupancy=current_d2_occupancy,
            tau=RECEPTOR_RESERVE.tau_d2_striatum,
            e_max=1.0
        )

        # Step 4: Motor improvement is the CHANGE in dopaminergic tone
        # Normalized to maximum possible improvement (baseline → full activation)
        max_possible_improvement = 1.0 - baseline_effect
        if max_possible_improvement <= 0:
            return 0.0

        actual_improvement = (current_effect - baseline_effect) / max_possible_improvement

        # Step 5: Scale to UPDRS percentage
        # Maximum UPDRS improvement typically ~50% with optimal levodopa
        max_updrs_improvement = 50.0
        motor_improvement = actual_improvement * max_updrs_improvement

        # Apply network fraction (only striatal D2 neurons contribute)
        motor_improvement *= (1.0 + NETWORK_PARAMS.dopamine_network_fraction)

        return min(50.0, max(0.0, motor_improvement))

    def _calculate_serotonin_increase(self) -> float:
        """
        Calculate synaptic serotonin increase (fluoxetine effect).

        Mechanistic cascade:
        1. Fluoxetine binds SERT (serotonin transporter)
        2. Blocks 5-HT reuptake → accumulation in synaptic cleft
        3. Increased 5-HT activates postsynaptic receptors
        4. But also 5-HT1A autoreceptors → negative feedback

        The negative feedback limits the initial serotonin increase.
        Full effect requires autoreceptor desensitization (weeks).

        References:
            - Wong DT et al. (2005) Nat Rev Drug Discov 4:764
            - Blier P, de Montigny C (1998) J Clin Psychopharmacol 18:2S
        """
        if not self.sert_transporters:
            return 10.0  # Baseline

        # Step 1: Get average SERT inhibition
        inhibitions = [t.transporter_inhibition for t in self.sert_transporters.values()]
        mean_inhibition = np.mean(inhibitions)

        # Step 2: Apply receptor reserve for SERT
        effective_inhibition = operational_model_effect(
            occupancy=mean_inhibition,
            tau=RECEPTOR_RESERVE.tau_sert,
            e_max=1.0
        )

        # Step 3: Calculate synaptic 5-HT increase
        # At steady state: [5-HT] = Release_rate / Clearance_rate
        # With SERT inhibition: Clearance = Clearance_baseline × (1 - inhibition)
        baseline_5ht = 10.0  # nM
        min_clearance = 0.1  # Cannot go to zero (diffusion, MAO)

        clearance_factor = 1.0 - effective_inhibition * 0.9  # 90% max reduction
        clearance_factor = max(min_clearance, clearance_factor)

        # 5-HT increase is inversely proportional to clearance
        raw_5ht_increase = baseline_5ht / clearance_factor

        # Step 4: Apply autoreceptor negative feedback
        # 5-HT1A autoreceptors reduce 5-HT release when activated
        # This limits acute increase to ~3-5x baseline
        autoreceptor_feedback = 1.0 - 0.5 * effective_inhibition  # 50% max reduction
        regulated_5ht = baseline_5ht + (raw_5ht_increase - baseline_5ht) * autoreceptor_feedback

        # Cap at physiological maximum (~80 nM, limited by autoreceptors)
        return min(80.0, max(baseline_5ht, regulated_5ht))

    def _calculate_beta_power(self) -> float:
        """
        Calculate beta oscillation power increase (diazepam/BZ effect).

        Mechanistic cascade:
        1. BZ binds GABA_A at α/γ interface (allosteric site)
        2. Potentiates GABA binding → enhanced Cl- currents
        3. Synchronized inhibition of cortical pyramidal neurons
        4. Rebound excitation generates beta oscillations (13-30 Hz)

        Beta signature is characteristic of benzodiazepines and correlates
        with anxiolysis. Different from propofol which causes slow delta.

        References:
            - Feshchenko VA et al. (2001) EEG patterns in midazolam sedation
            - Jensen O et al. (2005) Beta oscillations in cortex
        """
        # Step 1: Get average receptor modulation
        modulations = [r.modulation_factor for r in self.gaba_receptors.values()]
        mean_modulation = np.mean(modulations)

        # Step 2: Apply receptor reserve for GABAergic system
        # BZ effect on beta is through modulation, not direct agonism
        effective_modulation = operational_model_effect(
            occupancy=mean_modulation - 1.0,  # Excess modulation
            tau=RECEPTOR_RESERVE.tau_gaba_cortex,
            e_max=1.0
        )

        # Step 3: Calculate beta power increase
        # Beta emerges from synchronized inhibition → rebound excitation
        # Network model: beta ∝ GABAergic tone × E/I coupling
        beta_coupling = NETWORK_PARAMS.beta_gaba_coupling
        ei_factor = NETWORK_PARAMS.ei_ratio_cortex

        # Beta power scales with effective GABAergic modulation
        # Maximum increase observed clinically is ~50-80%
        max_beta_increase = 80.0  # % increase

        # Sigmoid relationship between modulation and beta
        # Moderate BZ doses give maximal beta, high doses reduce (sedation)
        beta_optimal_modulation = 0.3  # Peak beta at ~30% modulation increase
        if effective_modulation <= beta_optimal_modulation:
            # Rising phase
            beta_increase = max_beta_increase * (effective_modulation / beta_optimal_modulation)
        else:
            # Falling phase (oversedation reduces beta)
            overshoot = effective_modulation - beta_optimal_modulation
            beta_increase = max_beta_increase * (1.0 - overshoot)

        # Apply network coupling factor
        beta_increase *= beta_coupling * (1.0 + ei_factor)

        return max(0.0, min(max_beta_increase, beta_increase))

    def get_network_statistics(self) -> Dict:
        """Get network statistics for validation."""
        region_counts = {}
        for region in BrainRegion:
            region_counts[region.value] = self.neuron_regions.count(region)

        type_counts = {}
        for ntype in NeuronType:
            type_counts[ntype.value] = self.neuron_types.count(ntype)

        return {
            "total_neurons": len(self.neurons),
            "total_synapses": len(self.connections),
            "neurons_by_region": region_counts,
            "neurons_by_type": type_counts,
            "gaba_receptors": len(self.gaba_receptors),
            "nmda_receptors": len(self.nmda_receptors),
            "d2_receptors": len(self.d2_receptors),
            "sert_transporters": len(self.sert_transporters),
        }


def build_m1_network() -> BrainNetwork:
    """Build reduced network for M1 MacBook Air (8GB RAM)."""
    params = NetworkParameters(n_neurons_total=10_000)  # 10K neurons (calibration)
    network = BrainNetwork(params)
    network.build_network(neuron_model="izhikevich")  # Efficient model
    return network


def build_rtx3050_network() -> BrainNetwork:
    """Build full-scale network for RTX 3050 (16GB RAM)."""
    params = NetworkParameters(n_neurons_total=10_000_000)  # 10M neurons
    network = BrainNetwork(params)
    network.build_network(neuron_model="izhikevich")
    return network


if __name__ == "__main__":
    # Test network construction
    print("Testing brain network construction...\n")

    # Build M1-compatible network
    print("=" * 60)
    print("M1 MacBook Air Network (100K neurons)")
    print("=" * 60)
    network = build_m1_network()

    # Print statistics
    stats = network.get_network_statistics()
    print(f"\nNetwork Statistics:")
    print(f"  Total neurons: {stats['total_neurons']:,}")
    print(f"  Total synapses: {stats['total_synapses']:,}")
    print(f"  GABA_A receptors: {stats['gaba_receptors']:,}")
    print(f"  NMDA receptors: {stats['nmda_receptors']:,}")
    print(f"  Dopamine D2 receptors: {stats['d2_receptors']:,}")
    print(f"  SERT transporters: {stats['sert_transporters']:,}")

    # Test drug application
    print("\n" + "=" * 60)
    print("Testing Drug Effects")
    print("=" * 60)

    # Propofol
    effects = network.apply_drug("propofol", concentration=4.0, unit="uM")
    print(f"\n{effects['drug'].upper()} ({effects['concentration']} {effects['unit']}):")
    print(f"  Targets: {', '.join(effects['receptor_targets'])}")
    print(f"  EEG suppression: {effects['network_effects']['eeg_suppression_pct']:.1f}% (target: 60%)")

    # Ketamine
    effects = network.apply_drug("ketamine", concentration=5.0, unit="uM")
    print(f"\n{effects['drug'].upper()} ({effects['concentration']} {effects['unit']}):")
    print(f"  Targets: {', '.join(effects['receptor_targets'])}")
    print(f"  Gamma power increase: {effects['network_effects']['gamma_power_increase']:.2f}x")

    # Levodopa
    effects = network.apply_drug("levodopa", concentration=2.0, unit="uM")
    print(f"\n{effects['drug'].upper()} ({effects['concentration']} {effects['unit']}):")
    print(f"  Targets: {', '.join(effects['receptor_targets'])}")
    print(f"  Motor improvement: {effects['network_effects']['motor_improvement_pct']:.1f}% (target: 30-50%)")

    print("\nNetwork builder module test complete! ✓")
