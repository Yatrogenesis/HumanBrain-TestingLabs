"""
Neuron Models for Human Brain Simulation
=========================================

Implements various neuron models:
- Hodgkin-Huxley (detailed)
- Izhikevich (efficient)
- Leaky Integrate-and-Fire (simple)

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NeuronParameters:
    """Parameters for neuron models"""
    # Membrane properties
    C_m: float = 1.0  # Membrane capacitance (μF/cm²)
    g_L: float = 0.3  # Leak conductance (mS/cm²)
    E_L: float = -70.0  # Leak reversal potential (mV)

    # Spike properties
    V_thresh: float = -55.0  # Spike threshold (mV)
    V_reset: float = -70.0  # Reset potential (mV)
    V_spike: float = 20.0  # Spike peak (mV)

    # Time constants
    tau_ref: float = 2.0  # Refractory period (ms)


class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley neuron model with Na+, K+, and leak channels.

    Based on:
    Hodgkin AL, Huxley AF (1952) J Physiol 117:500-544
    """

    def __init__(self):
        # Maximal conductances (mS/cm²)
        self.g_Na = 120.0  # Sodium
        self.g_K = 36.0    # Potassium
        self.g_L = 0.3     # Leak

        # Reversal potentials (mV)
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.4

        # Membrane capacitance (μF/cm²)
        self.C_m = 1.0

        # State variables
        self.V = -65.0  # Membrane potential (mV)
        self.m = 0.05   # Na activation
        self.h = 0.6    # Na inactivation
        self.n = 0.32   # K activation

    def alpha_m(self, V: float) -> float:
        """Na+ activation rate"""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V: float) -> float:
        """Na+ deactivation rate"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V: float) -> float:
        """Na+ inactivation rate"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V: float) -> float:
        """Na+ de-inactivation rate"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V: float) -> float:
        """K+ activation rate"""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V: float) -> float:
        """K+ deactivation rate"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def step(self, I_ext: float, dt: float = 0.01) -> Tuple[float, bool]:
        """
        Integrate one timestep.

        Args:
            I_ext: External current (μA/cm²)
            dt: Timestep (ms)

        Returns:
            (voltage, spiked)
        """
        # Ion currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)

        # Membrane voltage
        dV = (I_ext - I_Na - I_K - I_L) / self.C_m
        self.V += dV * dt

        # Gating variables
        am, bm = self.alpha_m(self.V), self.beta_m(self.V)
        ah, bh = self.alpha_h(self.V), self.beta_h(self.V)
        an, bn = self.alpha_n(self.V), self.beta_n(self.V)

        self.m += (am * (1 - self.m) - bm * self.m) * dt
        self.h += (ah * (1 - self.h) - bh * self.h) * dt
        self.n += (an * (1 - self.n) - bn * self.n) * dt

        # Detect spike
        spiked = self.V > 0.0

        return self.V, spiked


class IzhikevichNeuron:
    """
    Izhikevich neuron model - efficient and versatile.

    Based on:
    Izhikevich EM (2003) IEEE Trans Neural Netw 14:1569-1572
    """

    def __init__(self, neuron_type: str = "regular_spiking"):
        """
        Args:
            neuron_type: 'regular_spiking', 'fast_spiking', 'intrinsic_bursting', 'chattering'
        """
        # Default parameters for regular spiking cortical neuron
        params = {
            "regular_spiking": (0.02, 0.2, -65.0, 8.0),
            "fast_spiking": (0.1, 0.2, -65.0, 2.0),
            "intrinsic_bursting": (0.02, 0.2, -55.0, 4.0),
            "chattering": (0.02, 0.2, -50.0, 2.0),
        }

        self.a, self.b, self.c, self.d = params[neuron_type]

        # State variables
        self.v = -65.0  # Membrane potential (mV)
        self.u = self.b * self.v  # Recovery variable

    def step(self, I_ext: float, dt: float = 0.5) -> Tuple[float, bool]:
        """
        Integrate one timestep.

        Args:
            I_ext: External current (pA)
            dt: Timestep (ms)

        Returns:
            (voltage, spiked)
        """
        spiked = False

        # Izhikevich equations
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I_ext)
        du = self.a * (self.b * self.v - self.u)

        self.v += dv * dt
        self.u += du * dt

        # Spike and reset
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            spiked = True

        return self.v, spiked


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron - simple and fast.

    Based on:
    Lapicque L (1907) J Physiol Pathol Gen 9:620-635
    """

    def __init__(self, params: NeuronParameters = None):
        self.params = params or NeuronParameters()

        # State variables
        self.V = self.params.E_L  # Membrane potential (mV)
        self.t_ref_remaining = 0.0  # Refractory time remaining (ms)

    def step(self, I_ext: float, dt: float = 0.1) -> Tuple[float, bool]:
        """
        Integrate one timestep.

        Args:
            I_ext: External current (nA)
            dt: Timestep (ms)

        Returns:
            (voltage, spiked)
        """
        spiked = False

        # Check refractory period
        if self.t_ref_remaining > 0:
            self.t_ref_remaining -= dt
            return self.V, False

        # Leak and input current
        tau_m = self.params.C_m / self.params.g_L
        dV = (-(self.V - self.params.E_L) + I_ext / self.params.g_L) / tau_m

        self.V += dV * dt

        # Spike and reset
        if self.V >= self.params.V_thresh:
            self.V = self.params.V_reset
            self.t_ref_remaining = self.params.tau_ref
            spiked = True

        return self.V, spiked


def simulate_neuron_population(
    n_neurons: int,
    model_type: str,
    duration: float,
    I_ext: np.ndarray,
    dt: float = 0.1
) -> Dict:
    """
    Simulate a population of neurons.

    Args:
        n_neurons: Number of neurons
        model_type: 'hodgkin_huxley', 'izhikevich', 'lif'
        duration: Simulation duration (ms)
        I_ext: External current (n_neurons x n_timesteps)
        dt: Timestep (ms)

    Returns:
        Dictionary with voltage traces and spike times
    """
    n_steps = int(duration / dt)

    # Create neurons
    if model_type == "hodgkin_huxley":
        neurons = [HodgkinHuxleyNeuron() for _ in range(n_neurons)]
    elif model_type == "izhikevich":
        neurons = [IzhikevichNeuron() for _ in range(n_neurons)]
    elif model_type == "lif":
        neurons = [LIFNeuron() for _ in range(n_neurons)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Storage
    voltage = np.zeros((n_neurons, n_steps))
    spike_times = [[] for _ in range(n_neurons)]

    # Simulate
    for t_idx in range(n_steps):
        for n_idx, neuron in enumerate(neurons):
            V, spiked = neuron.step(I_ext[n_idx, t_idx], dt)
            voltage[n_idx, t_idx] = V
            if spiked:
                spike_times[n_idx].append(t_idx * dt)

    return {
        "voltage": voltage,
        "spike_times": spike_times,
        "time": np.arange(0, duration, dt),
        "dt": dt
    }


if __name__ == "__main__":
    # Test all neuron models
    print("Testing neuron models...")

    # Test LIF
    lif = LIFNeuron()
    print(f"LIF initial voltage: {lif.V:.2f} mV")

    # Test Izhikevich
    izh = IzhikevichNeuron("regular_spiking")
    print(f"Izhikevich initial voltage: {izh.v:.2f} mV")

    # Test Hodgkin-Huxley
    hh = HodgkinHuxleyNeuron()
    print(f"Hodgkin-Huxley initial voltage: {hh.V:.2f} mV")

    print("\nAll neuron models initialized successfully! ✓")
