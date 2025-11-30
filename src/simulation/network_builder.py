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
        """Calculate EEG suppression percentage (propofol effect)."""
        suppressions = [r.get_suppression_percentage() for r in self.gaba_receptors.values()]
        return np.mean(suppressions)

    def _calculate_gamma_power(self) -> float:
        """Calculate gamma oscillation power increase (ketamine effect)."""
        gamma_factors = [r.get_gamma_power_increase() for r in self.nmda_receptors.values()]
        return np.mean(gamma_factors)

    def _calculate_motor_improvement(self) -> float:
        """Calculate motor function improvement (levodopa effect)."""
        if not self.d2_receptors:
            return 0.0

        baseline_da = 5.0  # nM (Parkinson's baseline)
        current_da = self.neurotransmitter_concentrations["dopamine"]

        improvements = [
            r.calculate_motor_improvement(baseline_da, current_da)
            for r in self.d2_receptors.values()
        ]
        return np.mean(improvements)

    def _calculate_serotonin_increase(self) -> float:
        """Calculate synaptic serotonin increase (fluoxetine effect)."""
        if not self.sert_transporters:
            return 10.0  # Baseline

        serotonins = [t.calculate_synaptic_serotonin() for t in self.sert_transporters.values()]
        return np.mean(serotonins)

    def _calculate_beta_power(self) -> float:
        """Calculate beta oscillation power increase (diazepam effect)."""
        # Benzodiazepines increase beta (13-30 Hz) power
        # CALIBRATED: reduced scaling for clinical alignment (Olkkola & Ahonen 2008)
        modulations = [r.modulation_factor for r in self.gaba_receptors.values()]
        beta_increase = 30.0 * (np.mean(modulations) - 1.0)  # Reduced from 100.0
        return beta_increase

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
