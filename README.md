# HumanBrain-TestingLabs 🧠💊

**Pharmacological Validation Framework for Computational Human Brain Models**

[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Private](https://img.shields.io/badge/Status-Private-red.svg)](https://github.com/Yatrogenesis/HumanBrain-TestingLabs)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%203050-green.svg)](EXPERIMENTAL_DESIGN.md)

---

## 🎯 Objetivo

Framework de validación rigurosa para simulaciones computacionales de efectos farmacológicos en cerebro humano mediante:

1. **Gold Standard Preliminar**: Fármacos con mecanismos extremadamente bien caracterizados
2. **Validación Ciega**: Protocolo doble-ciego con pre-registro
3. **Métricas Cuantitativas**: RMSE, correlación, error relativo vs datos clínicos reales
4. **Calibración Iterativa**: Ajuste de parámetros basado en discrepancias sistemáticas

---

## 🚨 DISCLAIMER

**ESTE ES UN PROYECTO DE INVESTIGACIÓN PRE-VALIDACIÓN**

- ❌ NO hacer claims clínicos sin peer review externo
- ❌ NO usar para decisiones médicas reales
- ❌ Resultados son SIMULACIONES, no ensayos clínicos
- ✅ Uso exclusivo: desarrollo y validación de modelos computacionales
- ✅ Requiere validación externa antes de publicación científica

---

## 📊 Fase 1: Fármacos Gold Standard

### Mecanismos Validados

| Fármaco | Mecanismo | Dosis Clínica | Efecto Medible | Referencia |
|---------|-----------|---------------|----------------|------------|
| **Propofol** | Agonista GABA_A | 2-6 μg/mL | Supresión EEG 60% | Brown et al. NEJM 2011 |
| **Ketamina** | Antagonista NMDA | 1-2 mg/kg IV | Ondas gamma 30-80 Hz | Sleigh et al. BJA 2014 |
| **Levodopa** | Precursor dopamina | 100-1000 mg/día | Mejora UPDRS 30-50% | Poewe et al. Nat Rev 2017 |
| **Fluoxetina** | SSRI (IC50 1 nM) | 20-80 mg/día | Latencia 2-4 semanas | Wong et al. NRDD 2005 |
| **Diazepam** | Modulador GABA_A | 2-10 mg | ↑ Beta (13-30 Hz) | Olkkola et al. CPK 2008 |

### Criterios de Aceptación

```
✅ Error relativo < 15%
✅ Correlación Pearson r > 0.85
✅ Reproducibilidad CV < 10%
```

---

## 🔬 Fase 2: Validación Ciega

### Protocolo Doble-Ciego

```
1. Investigador A: Selecciona 10 fármacos (identidad oculta)
2. Investigador B: Ejecuta simulaciones sin conocer fármacos
3. Investigador B: Predice efectos SOLO desde outputs
4. Revelación: Comparación con literatura médica
5. Análisis: Métricas de sesgo, varianza, exactitud
```

**Pre-registro obligatorio** en OSF.io antes de fase ciega.

---

## 💻 Hardware Requirements

### Opción A: M1 MacBook Air (Desarrollo)

```yaml
CPU: Apple M1 (8 cores: 4P+4E)
GPU: 7-core Metal 3
RAM: 8 GB
Capacidad: Simulaciones reducidas (10⁵-10⁶ neuronas)
Uso: Desarrollo, pruebas unitarias, regiones específicas
```

### Opción B: HP Victus 15 (Producción) ✅ RECOMENDADO

```yaml
GPU: NVIDIA RTX 3050 (4GB VRAM)
CPU: Intel i7-12700H (14 cores)
RAM: 16 GB
SSD: 256 GB
Capacidad: Cerebro humano completo (10⁷-10⁸ neuronas)
Uso: Simulaciones a gran escala, validación final
```

---

## 🛠️ Instalación

### 1. Clonar Repositorio

```bash
git clone https://github.com/Yatrogenesis/HumanBrain-TestingLabs.git
cd HumanBrain-TestingLabs
```

### 2. Configurar Entorno (CUDA para RTX 3050)

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Descargar Human Brain Model desde Zenodo

```bash
# Ejecutar script de descarga
python scripts/download_humanbrain_zenodo.py

# Verificar integridad
python scripts/verify_model_integrity.py
```

---

## 🚀 Uso Rápido

### Ejecutar Validación Gold Standard

```bash
# Validar Propofol (anestésico GABA_A)
python validate.py --drug propofol --dose 4.0 --output results/propofol_001.json

# Validar todos los fármacos gold standard
python validate_all_goldstandard.py --hardware rtx3050 --replicates 10
```

### Análisis de Resultados

```bash
# Calcular métricas agregadas
python analyze_results.py --input results/goldstandard/ --output metrics/summary.csv

# Generar visualizaciones
python plot_validation.py --input metrics/summary.csv --output figures/
```

---

## 📁 Estructura del Proyecto

```
HumanBrain-TestingLabs/
├── README.md
├── EXPERIMENTAL_DESIGN.md        # Diseño experimental completo
├── requirements.txt               # Dependencias Python
├── LICENSE
├── .gitignore
│
├── data/
│   ├── human_brain_model/        # Modelo desde Zenodo (gitignored)
│   ├── pharmacology/              # Perfiles farmacológicos
│   │   ├── propofol.json
│   │   ├── ketamine.json
│   │   └── ...
│   └── clinical_validation/       # Datos clínicos de referencia
│
├── src/
│   ├── simulation/                # Motor de simulación neuronal
│   │   ├── neuron_models.py      # Hodgkin-Huxley, Izhikevich, LIF
│   │   ├── synapse_models.py     # GABA, NMDA, dopamina, serotonina
│   │   └── network_builder.py    # Construcción de redes
│   │
│   ├── pharmacology/              # Módulos PK/PD
│   │   ├── pharmacokinetics.py   # PBPK, compartimentos
│   │   ├── pharmacodynamics.py   # Binding, eficacia
│   │   └── drug_effects.py       # Integración dosis→efecto
│   │
│   ├── validation/                # Framework de validación
│   │   ├── gold_standard.py      # Validación fase 1
│   │   ├── blind_testing.py      # Protocolo doble-ciego
│   │   └── metrics.py            # RMSE, correlación, error%
│   │
│   └── analysis/                  # Análisis estadístico
│       ├── statistical_tests.py
│       └── visualization.py
│
├── scripts/
│   ├── download_humanbrain_zenodo.py
│   ├── verify_model_integrity.py
│   ├── run_validation_suite.py
│   └── generate_blind_dataset.py
│
├── results/                       # Outputs de simulaciones (gitignored)
│   ├── goldstandard/
│   └── blind/
│
├── metrics/                       # Métricas calculadas
│   └── summary.csv
│
├── figures/                       # Visualizaciones
│   └── validation_plots.pdf
│
└── docs/
    ├── pharmacology_references.md
    ├── neural_models.md
    └── statistical_analysis_plan.md
```

---

## 📚 Referencias Clave

### Neurofarmacología
- Dayan P, Abbott LF (2001) *Theoretical Neuroscience* MIT Press
- Destexhe A, Sejnowski TJ (2009) "The Wilson-Cowan model" *Biol Cybern* 101:1

### Fármacos Gold Standard
- **Propofol**: Brown EN et al. (2011) NEJM 363:2638
- **Ketamina**: Sleigh JW et al. (2014) Br J Anaesth 113:i61
- **Levodopa**: Poewe W et al. (2017) Nat Rev Dis Primers 3:17013
- **Fluoxetina**: Wong DT et al. (2005) Nat Rev Drug Discov 4:764
- **Diazepam**: Olkkola KT et al. (2008) Clin Pharmacokinet 47:469

---

## 👤 Autor

**Francisco Molina Burgos**

- ORCID: [0009-0008-6093-8267](https://orcid.org/0009-0008-6093-8267)
- Email: pako.molina@gmail.com
- GitHub: [@Yatrogenesis](https://github.com/Yatrogenesis)

---

## 📄 Licencia

Dual licensed under MIT OR Apache-2.0

---

## ⚠️ Estado del Proyecto

🔴 **PRIVADO** - En desarrollo activo
🔬 **PRE-VALIDACIÓN** - Requiere revisión por pares antes de publicación
🧪 **INVESTIGACIÓN** - NO apto para uso clínico

---

**🚀 HumanBrain-TestingLabs - Rigorous Pharmacological Validation**
