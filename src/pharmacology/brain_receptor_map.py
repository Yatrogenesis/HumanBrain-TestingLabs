#!/usr/bin/env python3
"""
Comprehensive Brain Receptor Map
================================

Complete mapping of all CNS receptor systems and their interactions.
Based on vademecum data (28 receptor targets, 79 drugs).

This module defines:
1. All receptor types and their biophysical properties
2. Neurotransmitter systems and their effects
3. Inter-receptor interactions (cascade effects)
4. Regional brain distribution

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import numpy as np


# =============================================================================
# NEUROTRANSMITTER SYSTEMS
# =============================================================================

class NeurotransmitterSystem(Enum):
    """Major neurotransmitter systems in the brain."""
    GABAERGIC = "gabaergic"           # Inhibitory
    GLUTAMATERGIC = "glutamatergic"   # Excitatory
    DOPAMINERGIC = "dopaminergic"     # Reward/Motor
    SEROTONERGIC = "serotonergic"     # Mood/Anxiety
    NORADRENERGIC = "noradrenergic"   # Arousal/Attention
    CHOLINERGIC = "cholinergic"       # Memory/Attention
    OPIOIDERGIC = "opioidergic"       # Pain/Reward
    HISTAMINERGIC = "histaminergic"   # Arousal/Appetite


class ReceptorType(Enum):
    """Types of receptor signaling."""
    IONOTROPIC = "ionotropic"     # Fast, direct ion channel
    METABOTROPIC = "metabotropic" # Slow, G-protein coupled
    TRANSPORTER = "transporter"   # Reuptake transporter


class SignalDirection(Enum):
    """Direction of effect when receptor is activated."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


# =============================================================================
# RECEPTOR DEFINITION
# =============================================================================

@dataclass
class ReceptorProperties:
    """
    Complete properties of a brain receptor.

    This is the core data structure for the receptor map.
    """
    # Identification
    gene_symbol: str           # e.g., "GABRA1"
    chembl_id: str             # e.g., "CHEMBL2093869"
    full_name: str             # e.g., "GABA-A receptor alpha-1"

    # Classification
    neurotransmitter: NeurotransmitterSystem
    receptor_type: ReceptorType
    signal_direction: SignalDirection

    # Biophysical properties
    typical_ki_nM: float       # Typical binding affinity
    hill_coefficient: float = 1.0

    # Brain distribution (0-1 relative density)
    cortex_density: float = 0.5
    hippocampus_density: float = 0.5
    striatum_density: float = 0.5
    thalamus_density: float = 0.5
    brainstem_density: float = 0.5

    # Functional effects when activated (0-1)
    effects: Dict[str, float] = field(default_factory=dict)

    # Interactions with other receptors
    potentiates: List[str] = field(default_factory=list)    # Enhances these receptors
    inhibits: List[str] = field(default_factory=list)       # Suppresses these receptors
    requires: List[str] = field(default_factory=list)       # Needs these for full effect


# =============================================================================
# COMPLETE BRAIN RECEPTOR DATABASE
# =============================================================================

BRAIN_RECEPTORS: Dict[str, ReceptorProperties] = {
    # -------------------------------------------------------------------------
    # GABAERGIC SYSTEM (Inhibitory)
    # -------------------------------------------------------------------------
    "GABRA1": ReceptorProperties(
        gene_symbol="GABRA1",
        chembl_id="CHEMBL2093869",
        full_name="GABA-A receptor alpha-1",
        neurotransmitter=NeurotransmitterSystem.GABAERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=20.0,  # BZ site
        cortex_density=0.9,
        hippocampus_density=0.8,
        striatum_density=0.4,
        thalamus_density=0.9,
        brainstem_density=0.6,
        effects={
            "sedation": 0.8,
            "anxiolysis": 0.6,
            "amnesia": 0.5,
            "muscle_relaxation": 0.4,
            "anticonvulsant": 0.7,
        },
        potentiates=[],
        inhibits=["GRIN1", "GRIN2A", "GRIN2B"],  # Inhibits glutamate
    ),
    "GABRA2": ReceptorProperties(
        gene_symbol="GABRA2",
        chembl_id="CHEMBL2096902",
        full_name="GABA-A receptor alpha-2",
        neurotransmitter=NeurotransmitterSystem.GABAERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=25.0,
        cortex_density=0.7,
        hippocampus_density=0.9,  # High in hippocampus
        striatum_density=0.5,
        thalamus_density=0.6,
        brainstem_density=0.4,
        effects={
            "anxiolysis": 0.9,  # Primary anxiolytic subunit
            "sedation": 0.4,
            "amnesia": 0.3,
            "muscle_relaxation": 0.3,
        },
        inhibits=["GRIN1"],
    ),
    "GABRB2": ReceptorProperties(
        gene_symbol="GABRB2",
        chembl_id="CHEMBL2093868",
        full_name="GABA-A receptor beta-2",
        neurotransmitter=NeurotransmitterSystem.GABAERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=3500.0,  # Anesthetic site
        cortex_density=0.9,
        hippocampus_density=0.8,
        striatum_density=0.7,
        thalamus_density=0.9,
        brainstem_density=0.8,
        effects={
            "anesthesia": 0.95,
            "sedation": 0.9,
            "amnesia": 0.8,
            "anticonvulsant": 0.6,
        },
        inhibits=["GRIN1", "GRIN2A", "GRIN2B"],
    ),

    # -------------------------------------------------------------------------
    # GLUTAMATERGIC SYSTEM (Excitatory)
    # -------------------------------------------------------------------------
    "GRIN1": ReceptorProperties(
        gene_symbol="GRIN1",
        chembl_id="CHEMBL1907601",
        full_name="NMDA receptor NR1",
        neurotransmitter=NeurotransmitterSystem.GLUTAMATERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=1000.0,  # Ketamine site
        cortex_density=0.9,
        hippocampus_density=0.95,  # Critical for memory
        striatum_density=0.7,
        thalamus_density=0.8,
        brainstem_density=0.5,
        effects={
            "memory_formation": 0.9,
            "learning": 0.9,
            "synaptic_plasticity": 0.95,
            "pain_transmission": 0.7,
            "excitotoxicity": 0.8,  # Negative effect
        },
        potentiates=["DRD1"],  # Glutamate potentiates dopamine
        requires=["GRIN2A", "GRIN2B"],  # Needs NR2 subunits
    ),
    "GRIN2A": ReceptorProperties(
        gene_symbol="GRIN2A",
        chembl_id="CHEMBL1907605",
        full_name="NMDA receptor NR2A",
        neurotransmitter=NeurotransmitterSystem.GLUTAMATERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=500.0,
        cortex_density=0.9,
        hippocampus_density=0.9,
        striatum_density=0.6,
        thalamus_density=0.7,
        brainstem_density=0.4,
        effects={
            "synaptic_plasticity": 0.9,
            "ltp_induction": 0.85,  # Long-term potentiation
            "memory_consolidation": 0.8,
        },
    ),
    "GRIN2B": ReceptorProperties(
        gene_symbol="GRIN2B",
        chembl_id="CHEMBL1907600",
        full_name="NMDA receptor NR2B",
        neurotransmitter=NeurotransmitterSystem.GLUTAMATERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=500.0,
        cortex_density=0.8,
        hippocampus_density=0.85,
        striatum_density=0.7,
        thalamus_density=0.6,
        brainstem_density=0.5,
        effects={
            "pain_sensitization": 0.8,
            "depression_pathology": 0.7,  # Implicated in depression
            "synaptic_plasticity": 0.8,
        },
    ),

    # -------------------------------------------------------------------------
    # DOPAMINERGIC SYSTEM (Reward/Motor)
    # -------------------------------------------------------------------------
    "DRD1": ReceptorProperties(
        gene_symbol="DRD1",
        chembl_id="CHEMBL2056",
        full_name="Dopamine D1 receptor",
        neurotransmitter=NeurotransmitterSystem.DOPAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,  # Gs-coupled
        typical_ki_nM=100.0,
        cortex_density=0.8,  # Prefrontal cortex
        hippocampus_density=0.4,
        striatum_density=0.95,  # Very high in striatum
        thalamus_density=0.3,
        brainstem_density=0.2,
        effects={
            "reward": 0.9,
            "motivation": 0.8,
            "motor_initiation": 0.7,
            "working_memory": 0.7,
            "cognitive_flexibility": 0.6,
        },
        potentiates=["GRIN1"],  # D1 enhances NMDA
    ),
    "DRD2": ReceptorProperties(
        gene_symbol="DRD2",
        chembl_id="CHEMBL217",
        full_name="Dopamine D2 receptor",
        neurotransmitter=NeurotransmitterSystem.DOPAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,  # Gi-coupled
        typical_ki_nM=10.0,  # High affinity
        cortex_density=0.5,
        hippocampus_density=0.3,
        striatum_density=0.95,  # Very high
        thalamus_density=0.4,
        brainstem_density=0.8,  # VTA, substantia nigra
        effects={
            "motor_control": 0.9,
            "reward_modulation": 0.8,
            "prolactin_inhibition": 0.9,
            "psychosis_regulation": 0.9,  # Antipsychotic target
            "nausea_control": 0.7,
        },
        inhibits=["DRD1"],  # D2 opposes D1 in striatum
    ),
    "DRD3": ReceptorProperties(
        gene_symbol="DRD3",
        chembl_id="CHEMBL234",
        full_name="Dopamine D3 receptor",
        neurotransmitter=NeurotransmitterSystem.DOPAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=5.0,  # Very high affinity
        cortex_density=0.3,
        hippocampus_density=0.4,
        striatum_density=0.7,  # Nucleus accumbens
        thalamus_density=0.2,
        brainstem_density=0.3,
        effects={
            "reward_salience": 0.8,
            "addiction_vulnerability": 0.7,
            "impulse_control": 0.6,
        },
    ),
    "DRD4": ReceptorProperties(
        gene_symbol="DRD4",
        chembl_id="CHEMBL219",
        full_name="Dopamine D4 receptor",
        neurotransmitter=NeurotransmitterSystem.DOPAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=50.0,
        cortex_density=0.8,  # High in PFC
        hippocampus_density=0.5,
        striatum_density=0.3,
        thalamus_density=0.4,
        brainstem_density=0.2,
        effects={
            "attention": 0.8,  # ADHD relevance
            "novelty_seeking": 0.7,
            "cognitive_flexibility": 0.6,
        },
    ),

    # -------------------------------------------------------------------------
    # SEROTONERGIC SYSTEM (Mood/Anxiety)
    # -------------------------------------------------------------------------
    "HTR1A": ReceptorProperties(
        gene_symbol="HTR1A",
        chembl_id="CHEMBL214",
        full_name="Serotonin 5-HT1A receptor",
        neurotransmitter=NeurotransmitterSystem.SEROTONERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,  # Autoreceptor
        typical_ki_nM=2.0,  # High affinity
        cortex_density=0.8,
        hippocampus_density=0.9,  # Very high
        striatum_density=0.3,
        thalamus_density=0.4,
        brainstem_density=0.95,  # Raphe nuclei
        effects={
            "anxiolysis": 0.9,  # Primary anxiolytic
            "antidepressant": 0.8,
            "body_temperature": 0.6,
            "sexual_function": 0.5,
            "serotonin_release_inhibition": 0.9,  # Autoreceptor
        },
        inhibits=["SLC6A4"],  # Feedback inhibition
    ),
    "HTR2A": ReceptorProperties(
        gene_symbol="HTR2A",
        chembl_id="CHEMBL224",
        full_name="Serotonin 5-HT2A receptor",
        neurotransmitter=NeurotransmitterSystem.SEROTONERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,  # Gq-coupled
        typical_ki_nM=10.0,
        cortex_density=0.95,  # Very high in cortex
        hippocampus_density=0.6,
        striatum_density=0.4,
        thalamus_density=0.5,
        brainstem_density=0.3,
        effects={
            "hallucinations": 0.95,  # Psychedelic target
            "mood_modulation": 0.7,
            "platelet_aggregation": 0.6,
            "vasoconstriction": 0.5,
            "antipsychotic_target": 0.8,  # Atypical antipsychotics block this
        },
        potentiates=["GRIN1"],  # 5-HT2A enhances glutamate
    ),
    "HTR2C": ReceptorProperties(
        gene_symbol="HTR2C",
        chembl_id="CHEMBL225",
        full_name="Serotonin 5-HT2C receptor",
        neurotransmitter=NeurotransmitterSystem.SEROTONERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=15.0,
        cortex_density=0.7,
        hippocampus_density=0.8,
        striatum_density=0.6,
        thalamus_density=0.5,
        brainstem_density=0.4,
        effects={
            "appetite_suppression": 0.9,  # Weight loss drugs
            "mood_modulation": 0.6,
            "anxiety": 0.5,
            "dopamine_inhibition": 0.7,
        },
        inhibits=["DRD2", "DRD1"],  # 5-HT2C inhibits dopamine
    ),
    "SLC6A4": ReceptorProperties(
        gene_symbol="SLC6A4",
        chembl_id="CHEMBL228",
        full_name="Serotonin transporter (SERT)",
        neurotransmitter=NeurotransmitterSystem.SEROTONERGIC,
        receptor_type=ReceptorType.TRANSPORTER,
        signal_direction=SignalDirection.MODULATORY,
        typical_ki_nM=1.0,  # SSRIs are very potent
        cortex_density=0.8,
        hippocampus_density=0.9,
        striatum_density=0.5,
        thalamus_density=0.6,
        brainstem_density=0.95,  # Raphe nuclei
        effects={
            "serotonin_reuptake": 0.95,
            "antidepressant_target": 0.95,
            "anxiolysis": 0.7,
            "ocd_target": 0.8,
        },
        potentiates=["HTR1A", "HTR2A"],  # Blocking increases 5-HT
    ),

    # -------------------------------------------------------------------------
    # NORADRENERGIC SYSTEM (Arousal/Attention)
    # -------------------------------------------------------------------------
    "SLC6A2": ReceptorProperties(
        gene_symbol="SLC6A2",
        chembl_id="CHEMBL222",
        full_name="Norepinephrine transporter (NET)",
        neurotransmitter=NeurotransmitterSystem.NORADRENERGIC,
        receptor_type=ReceptorType.TRANSPORTER,
        signal_direction=SignalDirection.MODULATORY,
        typical_ki_nM=5.0,
        cortex_density=0.8,
        hippocampus_density=0.7,
        striatum_density=0.4,
        thalamus_density=0.6,
        brainstem_density=0.95,  # Locus coeruleus
        effects={
            "attention": 0.9,
            "arousal": 0.85,
            "antidepressant_target": 0.8,
            "pain_modulation": 0.6,
            "blood_pressure": 0.5,
        },
        potentiates=["ADRA1A", "ADRA2A", "ADRB1"],
    ),
    "SLC6A3": ReceptorProperties(
        gene_symbol="SLC6A3",
        chembl_id="CHEMBL238",
        full_name="Dopamine transporter (DAT)",
        neurotransmitter=NeurotransmitterSystem.DOPAMINERGIC,
        receptor_type=ReceptorType.TRANSPORTER,
        signal_direction=SignalDirection.MODULATORY,
        typical_ki_nM=200.0,
        cortex_density=0.6,
        hippocampus_density=0.3,
        striatum_density=0.95,  # Very high
        thalamus_density=0.3,
        brainstem_density=0.8,  # VTA
        effects={
            "dopamine_reuptake": 0.95,
            "stimulant_target": 0.95,  # Cocaine, amphetamine
            "adhd_target": 0.9,
            "reward": 0.8,
            "addiction_vulnerability": 0.9,
        },
        potentiates=["DRD1", "DRD2"],
    ),
    "ADRA1A": ReceptorProperties(
        gene_symbol="ADRA1A",
        chembl_id="CHEMBL229",
        full_name="Alpha-1A adrenergic receptor",
        neurotransmitter=NeurotransmitterSystem.NORADRENERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=50.0,
        cortex_density=0.7,
        hippocampus_density=0.5,
        striatum_density=0.3,
        thalamus_density=0.6,
        brainstem_density=0.4,
        effects={
            "vasoconstriction": 0.9,
            "smooth_muscle_contraction": 0.8,
            "arousal": 0.6,
            "mydriasis": 0.7,
        },
    ),
    "ADRA2A": ReceptorProperties(
        gene_symbol="ADRA2A",
        chembl_id="CHEMBL1867",
        full_name="Alpha-2A adrenergic receptor",
        neurotransmitter=NeurotransmitterSystem.NORADRENERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,  # Presynaptic autoreceptor
        typical_ki_nM=10.0,
        cortex_density=0.8,  # Prefrontal
        hippocampus_density=0.6,
        striatum_density=0.4,
        thalamus_density=0.5,
        brainstem_density=0.9,
        effects={
            "sedation": 0.8,  # Clonidine, dexmedetomidine
            "analgesia": 0.7,
            "hypotension": 0.8,
            "attention_enhancement": 0.7,  # At low doses
            "norepinephrine_inhibition": 0.9,  # Autoreceptor
        },
        inhibits=["SLC6A2"],  # Reduces NE release
    ),
    "ADRB1": ReceptorProperties(
        gene_symbol="ADRB1",
        chembl_id="CHEMBL213",
        full_name="Beta-1 adrenergic receptor",
        neurotransmitter=NeurotransmitterSystem.NORADRENERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=100.0,
        cortex_density=0.5,
        hippocampus_density=0.4,
        striatum_density=0.3,
        thalamus_density=0.4,
        brainstem_density=0.6,
        effects={
            "heart_rate_increase": 0.95,
            "cardiac_output": 0.9,
            "memory_consolidation": 0.6,  # Stress memories
        },
    ),
    "ADRB2": ReceptorProperties(
        gene_symbol="ADRB2",
        chembl_id="CHEMBL210",
        full_name="Beta-2 adrenergic receptor",
        neurotransmitter=NeurotransmitterSystem.NORADRENERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=150.0,
        cortex_density=0.4,
        hippocampus_density=0.5,
        striatum_density=0.3,
        thalamus_density=0.4,
        brainstem_density=0.5,
        effects={
            "bronchodilation": 0.95,
            "vasodilation": 0.7,
            "tremor": 0.6,
        },
    ),

    # -------------------------------------------------------------------------
    # OPIOIDERGIC SYSTEM (Pain/Reward)
    # -------------------------------------------------------------------------
    "OPRM1": ReceptorProperties(
        gene_symbol="OPRM1",
        chembl_id="CHEMBL233",
        full_name="Mu opioid receptor",
        neurotransmitter=NeurotransmitterSystem.OPIOIDERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=1.0,  # Morphine
        cortex_density=0.5,
        hippocampus_density=0.4,
        striatum_density=0.7,  # Nucleus accumbens
        thalamus_density=0.8,  # Pain pathways
        brainstem_density=0.95,  # PAG, respiratory centers
        effects={
            "analgesia": 0.95,
            "euphoria": 0.9,
            "respiratory_depression": 0.9,
            "constipation": 0.8,
            "miosis": 0.8,
            "physical_dependence": 0.9,
        },
        potentiates=["DRD1", "DRD2"],  # Opioids enhance dopamine
        inhibits=["GABRA1"],  # Disinhibition
    ),
    "OPRD1": ReceptorProperties(
        gene_symbol="OPRD1",
        chembl_id="CHEMBL236",
        full_name="Delta opioid receptor",
        neurotransmitter=NeurotransmitterSystem.OPIOIDERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=5.0,
        cortex_density=0.7,
        hippocampus_density=0.6,
        striatum_density=0.5,
        thalamus_density=0.4,
        brainstem_density=0.5,
        effects={
            "analgesia": 0.7,
            "antidepressant": 0.6,
            "anxiolysis": 0.5,
            "seizure_threshold": 0.4,  # Can lower
        },
    ),
    "OPRK1": ReceptorProperties(
        gene_symbol="OPRK1",
        chembl_id="CHEMBL237",
        full_name="Kappa opioid receptor",
        neurotransmitter=NeurotransmitterSystem.OPIOIDERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,
        typical_ki_nM=10.0,
        cortex_density=0.6,
        hippocampus_density=0.5,
        striatum_density=0.6,
        thalamus_density=0.5,
        brainstem_density=0.6,
        effects={
            "analgesia": 0.7,
            "dysphoria": 0.8,  # Opposite of mu
            "diuresis": 0.6,
            "hallucinations": 0.5,  # Salvinorin A
        },
        inhibits=["OPRM1"],  # Kappa opposes mu reward
    ),

    # -------------------------------------------------------------------------
    # CHOLINERGIC SYSTEM (Memory/Attention)
    # -------------------------------------------------------------------------
    "CHRM1": ReceptorProperties(
        gene_symbol="CHRM1",
        chembl_id="CHEMBL216",
        full_name="Muscarinic M1 receptor",
        neurotransmitter=NeurotransmitterSystem.CHOLINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=50.0,
        cortex_density=0.9,  # Very high
        hippocampus_density=0.9,  # Critical for memory
        striatum_density=0.8,
        thalamus_density=0.5,
        brainstem_density=0.3,
        effects={
            "memory_enhancement": 0.9,
            "cognitive_function": 0.8,
            "attention": 0.7,
            "salivation": 0.6,
        },
        potentiates=["GRIN1"],  # ACh enhances glutamate
    ),
    "CHRM2": ReceptorProperties(
        gene_symbol="CHRM2",
        chembl_id="CHEMBL211",
        full_name="Muscarinic M2 receptor",
        neurotransmitter=NeurotransmitterSystem.CHOLINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,  # Presynaptic
        typical_ki_nM=30.0,
        cortex_density=0.6,
        hippocampus_density=0.7,
        striatum_density=0.5,
        thalamus_density=0.6,
        brainstem_density=0.8,
        effects={
            "bradycardia": 0.9,
            "smooth_muscle_contraction": 0.7,
            "acetylcholine_inhibition": 0.8,  # Autoreceptor
        },
    ),
    "CHRNA4": ReceptorProperties(
        gene_symbol="CHRNA4",
        chembl_id="CHEMBL3884",
        full_name="Nicotinic alpha-4 receptor",
        neurotransmitter=NeurotransmitterSystem.CHOLINERGIC,
        receptor_type=ReceptorType.IONOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=20.0,
        cortex_density=0.7,
        hippocampus_density=0.6,
        striatum_density=0.5,
        thalamus_density=0.9,  # Very high
        brainstem_density=0.8,
        effects={
            "attention": 0.9,
            "cognitive_enhancement": 0.8,
            "nicotine_dependence": 0.95,
            "dopamine_release": 0.7,  # Indirect
        },
        potentiates=["DRD1", "DRD2", "SLC6A3"],  # Nicotine releases DA
    ),

    # -------------------------------------------------------------------------
    # HISTAMINERGIC SYSTEM (Arousal/Appetite)
    # -------------------------------------------------------------------------
    "HRH1": ReceptorProperties(
        gene_symbol="HRH1",
        chembl_id="CHEMBL231",
        full_name="Histamine H1 receptor",
        neurotransmitter=NeurotransmitterSystem.HISTAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.EXCITATORY,
        typical_ki_nM=20.0,
        cortex_density=0.7,
        hippocampus_density=0.6,
        striatum_density=0.4,
        thalamus_density=0.8,
        brainstem_density=0.5,
        effects={
            "arousal": 0.9,  # Blocking causes sedation
            "wakefulness": 0.85,
            "allergic_response": 0.9,
            "appetite_stimulation": 0.6,
        },
        potentiates=["GRIN1"],  # Histamine enhances glutamate
    ),
    "HRH3": ReceptorProperties(
        gene_symbol="HRH3",
        chembl_id="CHEMBL264",
        full_name="Histamine H3 receptor",
        neurotransmitter=NeurotransmitterSystem.HISTAMINERGIC,
        receptor_type=ReceptorType.METABOTROPIC,
        signal_direction=SignalDirection.INHIBITORY,  # Autoreceptor
        typical_ki_nM=5.0,
        cortex_density=0.8,
        hippocampus_density=0.7,
        striatum_density=0.9,  # Very high in striatum
        thalamus_density=0.6,
        brainstem_density=0.5,
        effects={
            "histamine_release_inhibition": 0.9,
            "cognitive_enhancement": 0.7,  # Blocking H3 enhances cognition
            "wakefulness": 0.6,
            "appetite_suppression": 0.5,
        },
        inhibits=["HRH1"],  # Autoreceptor effect
    ),
}


# =============================================================================
# CASCADE EFFECT MODEL
# =============================================================================

@dataclass
class CascadeEffect:
    """
    Represents a cascade effect between receptor systems.

    When source_receptor is activated, it affects target_receptor
    with a given strength and time delay.
    """
    source_receptor: str
    target_receptor: str
    effect_type: str  # "potentiation" or "inhibition"
    strength: float   # 0-1, how strong the effect is
    delay_ms: float   # Time delay for the cascade


# Define the cascade network
CASCADE_NETWORK: List[CascadeEffect] = [
    # GABA inhibits glutamate
    CascadeEffect("GABRA1", "GRIN1", "inhibition", 0.8, 10),
    CascadeEffect("GABRA1", "GRIN2A", "inhibition", 0.7, 10),
    CascadeEffect("GABRB2", "GRIN1", "inhibition", 0.9, 5),  # Faster for anesthetics

    # Glutamate-Dopamine interactions
    CascadeEffect("GRIN1", "DRD1", "potentiation", 0.7, 50),
    CascadeEffect("DRD1", "GRIN1", "potentiation", 0.5, 100),

    # D1-D2 opposition in striatum
    CascadeEffect("DRD2", "DRD1", "inhibition", 0.6, 20),

    # Serotonin-Dopamine interactions
    CascadeEffect("HTR2C", "DRD2", "inhibition", 0.7, 100),
    CascadeEffect("HTR2C", "DRD1", "inhibition", 0.6, 100),
    CascadeEffect("HTR2A", "GRIN1", "potentiation", 0.5, 50),

    # 5-HT1A autoreceptor feedback
    CascadeEffect("SLC6A4", "HTR1A", "potentiation", 0.8, 1000),  # Slow (SSRI delay)
    CascadeEffect("HTR1A", "SLC6A4", "inhibition", 0.6, 500),

    # Opioid-Dopamine reward pathway
    CascadeEffect("OPRM1", "DRD1", "potentiation", 0.8, 200),
    CascadeEffect("OPRM1", "DRD2", "potentiation", 0.7, 200),
    CascadeEffect("OPRM1", "GABRA1", "inhibition", 0.5, 100),  # Disinhibition

    # Kappa opposes mu
    CascadeEffect("OPRK1", "OPRM1", "inhibition", 0.6, 50),

    # Cholinergic-Glutamate enhancement
    CascadeEffect("CHRM1", "GRIN1", "potentiation", 0.6, 50),
    CascadeEffect("CHRNA4", "DRD1", "potentiation", 0.7, 30),
    CascadeEffect("CHRNA4", "SLC6A3", "potentiation", 0.6, 30),

    # Noradrenergic modulation
    CascadeEffect("SLC6A2", "ADRA2A", "potentiation", 0.8, 100),
    CascadeEffect("ADRA2A", "SLC6A2", "inhibition", 0.7, 200),

    # Histamine arousal pathway
    CascadeEffect("HRH1", "GRIN1", "potentiation", 0.5, 100),
    CascadeEffect("HRH3", "HRH1", "inhibition", 0.7, 50),
]


class BrainReceptorModel:
    """
    Complete brain receptor interaction model.

    Simulates how drugs affect receptor systems and how those
    effects cascade through the brain.
    """

    def __init__(self):
        self.receptors = BRAIN_RECEPTORS
        self.cascades = CASCADE_NETWORK
        self.receptor_states: Dict[str, float] = {r: 0.0 for r in self.receptors}

    def bind_drug(self, receptor_id: str, concentration_uM: float,
                  ki_nM: float = None) -> Dict:
        """
        Bind a drug to a receptor and calculate cascade effects.

        Returns dict with direct and cascade effects.
        """
        if receptor_id not in self.receptors:
            raise ValueError(f"Unknown receptor: {receptor_id}")

        receptor = self.receptors[receptor_id]

        # Use provided Ki or default
        ki = ki_nM if ki_nM else receptor.typical_ki_nM
        ic50_uM = ki / 1000.0

        # Calculate occupancy
        occupancy = concentration_uM / (ic50_uM + concentration_uM)
        self.receptor_states[receptor_id] = occupancy

        # Calculate direct effects
        direct_effects = {
            effect: strength * occupancy
            for effect, strength in receptor.effects.items()
        }

        # Calculate cascade effects
        cascade_effects = self._calculate_cascades(receptor_id, occupancy)

        return {
            "receptor": receptor_id,
            "occupancy": occupancy,
            "direct_effects": direct_effects,
            "cascade_effects": cascade_effects,
            "brain_regions": {
                "cortex": receptor.cortex_density * occupancy,
                "hippocampus": receptor.hippocampus_density * occupancy,
                "striatum": receptor.striatum_density * occupancy,
                "thalamus": receptor.thalamus_density * occupancy,
                "brainstem": receptor.brainstem_density * occupancy,
            }
        }

    def _calculate_cascades(self, source: str, source_occupancy: float) -> Dict[str, float]:
        """Calculate cascade effects from a receptor activation."""
        effects = {}

        for cascade in self.cascades:
            if cascade.source_receptor == source:
                target = cascade.target_receptor

                # Calculate cascade strength
                if cascade.effect_type == "potentiation":
                    effect = cascade.strength * source_occupancy
                else:  # inhibition
                    effect = -cascade.strength * source_occupancy

                if target in effects:
                    effects[target] += effect
                else:
                    effects[target] = effect

        return effects

    def get_receptor_info(self, receptor_id: str) -> ReceptorProperties:
        """Get full information about a receptor."""
        return self.receptors.get(receptor_id)

    def get_receptors_by_system(self, system: NeurotransmitterSystem) -> List[str]:
        """Get all receptors in a neurotransmitter system."""
        return [
            r_id for r_id, r in self.receptors.items()
            if r.neurotransmitter == system
        ]

    def get_receptor_summary(self) -> Dict:
        """Get summary of all receptor systems."""
        summary = {}
        for system in NeurotransmitterSystem:
            receptors = self.get_receptors_by_system(system)
            summary[system.value] = {
                "receptors": receptors,
                "count": len(receptors),
            }
        return summary


def demo_brain_map():
    """Demonstrate the brain receptor map."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BRAIN RECEPTOR MAP")
    print("=" * 80)

    model = BrainReceptorModel()

    # Summary of receptor systems
    print("\n1. RECEPTOR SYSTEMS SUMMARY")
    print("-" * 60)
    summary = model.get_receptor_summary()
    for system, info in summary.items():
        print(f"\n{system.upper()}: {info['count']} receptors")
        for r in info['receptors']:
            receptor = model.get_receptor_info(r)
            print(f"  - {r}: {receptor.full_name}")

    # Test drug binding
    print("\n\n2. DRUG BINDING SIMULATION")
    print("-" * 60)

    # Simulate diazepam binding
    result = model.bind_drug("GABRA1", concentration_uM=0.2)
    print(f"\nDiazepam binding to GABRA1 at 0.2 uM:")
    print(f"  Occupancy: {result['occupancy']:.1%}")
    print(f"  Direct effects:")
    for effect, value in result['direct_effects'].items():
        print(f"    - {effect}: {value:.1%}")
    print(f"  Cascade effects:")
    for target, effect in result['cascade_effects'].items():
        direction = "+" if effect > 0 else ""
        print(f"    - {target}: {direction}{effect:.1%}")

    # Simulate fluoxetine binding (SSRI)
    result = model.bind_drug("SLC6A4", concentration_uM=0.1, ki_nM=1.0)
    print(f"\nFluoxetine binding to SERT at 0.1 uM:")
    print(f"  Occupancy: {result['occupancy']:.1%}")
    print(f"  Direct effects:")
    for effect, value in result['direct_effects'].items():
        print(f"    - {effect}: {value:.1%}")
    print(f"  Cascade effects:")
    for target, effect in result['cascade_effects'].items():
        direction = "+" if effect > 0 else ""
        print(f"    - {target}: {direction}{effect:.1%}")

    print("\n" + "=" * 80)
    print(f"Total receptors mapped: {len(BRAIN_RECEPTORS)}")
    print(f"Total cascade connections: {len(CASCADE_NETWORK)}")
    print("=" * 80)


if __name__ == "__main__":
    demo_brain_map()
