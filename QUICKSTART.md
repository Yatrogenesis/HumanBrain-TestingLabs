# QuickStart Guide 🚀

**Get running in 5 minutes on M1 MacBook Air or RTX 3050!**

---

## 1. Clone Repository

```bash
git clone https://github.com/Yatrogenesis/HumanBrain-TestingLabs.git
cd HumanBrain-TestingLabs
```

---

## 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

**Expected time:** ~2-3 minutes

---

## 3. Test Installation

Run all module tests to verify everything works:

```bash
# Test neuron models
python3 src/simulation/neuron_models.py

# Test receptor models
python3 src/simulation/synapse_models.py

# Test pharmacokinetics
python3 src/pharmacology/pharmacokinetics.py
```

**Expected output:** All tests should pass with ✓

---

## 4. Run Your First Validation

### Example 1: Propofol (Anesthetic)

```bash
python3 validate.py --drug propofol --dose 2.0 --route IV --hardware m1
```

**What this does:**
- Simulates GABA_A receptor modulation
- Calculates brain concentration via PK model
- Predicts EEG suppression (target: 60%)
- Validates against clinical data (Brown et al. 2011)

**Expected time:** 30-60 seconds on M1

**Example output:**
```
================================================================================
Validating: Propofol
Mechanism: GABA_A positive allosteric modulator
Reference: Brown EN et al. (2011) NEJM 363:2638
================================================================================

Step 1: Simulating pharmacokinetics...
  Peak brain concentration: 4.13 μM at 0.10 h

Step 2: Applying drug to brain network...
  Receptor targets: GABA_A

Step 3: Validating against clinical data...
  Simulated EEG suppression: 57.9%
  Target: 60.0% ± 10.0%
  Error: 3.5%
  ✓ PASS

================================================================================
Results saved to: results/goldstandard/propofol_20251130_040500.json
================================================================================

✓ VALIDATION PASSED
```

---

### Example 2: Ketamine (Anesthetic/Antidepressant)

```bash
python3 validate.py --drug ketamine --dose 2.0 --route IM --hardware m1
```

**Validates:** NMDA antagonism → gamma oscillation increase (30-80 Hz)

---

### Example 3: Levodopa (Parkinson's Treatment)

```bash
python3 validate.py --drug levodopa --dose 100 --route oral --hardware m1
```

**Validates:** Dopamine increase → motor function improvement (UPDRS 30-50%)

---

## 5. Run Complete Validation Suite

Test all 5 gold standard drugs:

```bash
python3 validate_all_goldstandard.py --hardware m1 --replicates 1
```

**Drugs tested:**
1. Propofol (GABA_A agonist)
2. Ketamine (NMDA antagonist)
3. Levodopa (dopamine precursor)
4. Fluoxetine (SSRI)
5. Diazepam (benzodiazepine)

**Expected time:** ~3-5 minutes on M1

**Example output:**
```
================================================================================
GOLD STANDARD DRUG VALIDATION SUITE
Hardware: M1
Replicates per drug: 1
Total validations: 5
================================================================================

Testing: PROPOFOL - GABA_A agonist anesthetic
  Replicate 1/1...
    Error: 3.50% - ✓ PASS

Testing: KETAMINE - NMDA antagonist anesthetic
  Replicate 1/1...
    Error: 0.00% - ✓ PASS

Testing: LEVODOPA - Dopamine precursor for Parkinson's
  Replicate 1/1...
    Error: 14.30% - ✓ PASS

Testing: FLUOXETINE - SSRI antidepressant
  Replicate 1/1...
    Error: 0.00% - ✓ PASS

Testing: DIAZEPAM - Benzodiazepine anxiolytic
  Replicate 1/1...
    Error: 0.00% - ✓ PASS

================================================================================
VALIDATION SUITE COMPLETE
================================================================================

Total validations: 5
Passed: 5 (100.0%)
Failed: 0

Error Statistics:
  Mean: 3.56%
  Median: 0.00%
  Range: 0.00% - 14.30%

Results saved to: results/goldstandard/suite_20251130_040600
Summary: results/goldstandard/suite_20251130_040600/summary.json

✓ ALL VALIDATIONS PASSED
```

---

## 6. Analyze Results

Results are saved as JSON files in `results/goldstandard/`:

```bash
# View latest results
cat results/goldstandard/propofol_*.json | python3 -m json.tool
```

**Example JSON:**
```json
{
  "drug_name": "propofol",
  "dose_mg": 140.0,
  "route": "IV",
  "timestamp": "2025-11-30T04:05:00.123456",
  "hardware": "m1",
  "pk_profile": {
    "peak_brain_concentration_uM": 4.13,
    "peak_time_hours": 0.10
  },
  "network_effects": {
    "eeg_suppression_pct": 57.9
  },
  "clinical_targets": {
    "eeg_suppression_pct": 60.0,
    "reference": "Brown EN et al. (2011) NEJM 363:2638"
  },
  "validation_metrics": {
    "simulated_eeg_suppression_pct": 57.9,
    "target_eeg_suppression_pct": 60.0,
    "error_pct": 3.5,
    "within_tolerance": true
  }
}
```

---

## 7. Understanding the Results

### Validation Criteria

✅ **PASS**: Error < 15% relative to clinical target
✗ **FAIL**: Error ≥ 15%

### What Each Drug Tests

| Drug | Target | Clinical Effect | Acceptance Range |
|------|--------|----------------|------------------|
| **Propofol** | GABA_A | EEG suppression 60% | 51-69% |
| **Ketamine** | NMDA | Gamma power 2.5x | 2.1-2.9x |
| **Levodopa** | D2 | UPDRS improve 40% | 34-46% |
| **Fluoxetine** | SERT | 5-HT increase 50 nM | 43-58 nM |
| **Diazepam** | GABA_A | Beta power +40% | 34-46% |

---

## 8. Hardware Considerations

### M1 MacBook Air (Recommended for Testing)
- **Network size:** 100,000 neurons
- **RAM usage:** ~2-3 GB
- **Speed:** ~30-60 sec per drug
- **Best for:** Development, testing, single drugs

### HP Victus 15 RTX 3050 (Production)
- **Network size:** 10,000,000 neurons (100x larger)
- **RAM usage:** ~8-12 GB
- **Speed:** ~2-5 min per drug
- **Best for:** Full validation, batch runs

To use RTX 3050:
```bash
python3 validate.py --drug propofol --dose 2.0 --route IV --hardware rtx3050
```

---

## 9. Next Steps

### Run Individual Modules

```bash
# Test individual receptor models
python3 -c "from src.simulation.synapse_models import GABAaReceptor; r = GABAaReceptor(); r.bind_drug(4.0, 'propofol'); print(f'Suppression: {r.get_suppression_percentage():.1f}%')"

# Test PK for specific drug
python3 -c "from src.pharmacology.pharmacokinetics import simulate_pk_profile, RouteOfAdministration; import matplotlib.pyplot as plt; r = simulate_pk_profile('propofol', 140, RouteOfAdministration.IV); plt.plot(r['time_hours'], r['brain_concentration_uM']); plt.show()"
```

### Customize Parameters

Edit `src/simulation/network_builder.py` to change:
- Network size
- Connection probabilities
- Receptor densities

### Add New Drugs

1. Add PK parameters to `DrugPKDatabase` in `pharmacokinetics.py`
2. Add receptor model to `synapse_models.py`
3. Add clinical targets to `DrugValidator.VALIDATION_TARGETS` in `validate.py`

---

## 10. Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Memory Errors on M1

**Problem:** `MemoryError: Unable to allocate array`

**Solution:** Reduce network size in `network_builder.py`:
```python
NetworkParameters(n_neurons_total=50_000)  # Instead of 100,000
```

### Slow Performance

**Problem:** Validation takes >5 minutes per drug

**Solution:**
1. Close other applications
2. Reduce replicates: `--replicates 1`
3. Use smaller network (see above)

---

## 11. Getting Help

- **Issues:** https://github.com/Yatrogenesis/HumanBrain-TestingLabs/issues
- **Email:** pako.molina@gmail.com
- **ORCID:** 0009-0008-6093-8267

---

## 12. Citation

If you use this framework in your research:

```bibtex
@software{molina2025humanbrain,
  author = {Molina Burgos, Francisco},
  title = {HumanBrain-TestingLabs: Pharmacological Validation Framework},
  year = {2025},
  url = {https://github.com/Yatrogenesis/HumanBrain-TestingLabs},
  note = {Pre-validation research code - NOT for clinical use}
}
```

---

**⚠️ IMPORTANT DISCLAIMER**

This is a PRE-VALIDATION research framework:
- ❌ NOT validated for clinical use
- ❌ NOT for medical decisions
- ✅ For computational model development only
- ✅ Requires external peer review before publication

---

**🎉 You're ready to validate brain simulations!**

Start with: `python3 validate.py --drug propofol --dose 2.0 --route IV --hardware m1`
