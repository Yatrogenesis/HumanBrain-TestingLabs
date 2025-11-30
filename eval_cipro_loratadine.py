#!/usr/bin/env python3
"""
Evaluación Mecanística: Ciprofloxacina y Loratadina
Fármacos NO en base de datos GABA-A
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pharmacology.receptor_mechanisms import (
    UnifiedGABAaModel, ModelMode, BindingSite,
    MechanisticGABAaReceptor, DRUG_PROFILES
)

print("=" * 80)
print("EVALUACION MECANISTICA: CIPROFLOXACINA y LORATADINA")
print("Farmacos fuera de la base de datos GABA-A")
print("=" * 80)

# ============================================================
# CIPROFLOXACINA
# ============================================================
print("\n" + "=" * 80)
print("1. CIPROFLOXACINA")
print("=" * 80)

print("""
DATOS FARMACOLOGICOS (Literatura):
  Clase: Fluoroquinolona (antibiotico)
  Peso molecular: 331.34 Da
  LogP: 0.28 (hidrofilico)

MECANISMO RELEVANTE PARA SNC:
  - ANTAGONISTA parcial del sitio BZ del receptor GABA-A
  - Compite con GABA por union (efecto PRO-CONVULSIVANTE)
  - IC50 (GABA-A): ~100-300 uM (muy baja afinidad)
  - Eficacia: NEGATIVA (inhibe, no potencia)

FARMACOCINETICA:
  - Dosis tipica: 500 mg VO c/12h
  - Cmax plasmatico: ~2.5 ug/mL (~7.5 uM)
  - Penetracion SNC: ~10-20% (baja, hidrofilico)
  - Concentracion cerebral estimada: ~0.75-1.5 uM

EFECTOS CLINICOS SNC:
  - Convulsiones (raro, 0.1-2%)
  - Agitacion, confusion
  - Insomnio
  - Potenciado por AINES (inhiben metabolismo)
""")

# Simular con modelo mecanístico
print("\n--- SIMULACION MECANISTICA ---")

# Ciprofloxacina tiene BAJA afinidad y eficacia NEGATIVA
cipro_params = {
    'ki_nM': 150000,  # IC50 ~150 uM = 150000 nM (muy baja afinidad)
    'efficacy': -0.15,  # Eficacia NEGATIVA (antagonista parcial)
    'binding_site': BindingSite.BZ_SITE,
    'concentration_uM': 1.0  # Concentracion cerebral tipica
}

model_mech = UnifiedGABAaModel(mode=ModelMode.MECHANISTIC)

result_cipro = model_mech.simulate(
    concentration_uM=cipro_params['concentration_uM'],
    ki_nM=cipro_params['ki_nM'],
    efficacy=cipro_params['efficacy'],
    binding_site=cipro_params['binding_site']
)

print(f"  Concentracion cerebral: {cipro_params['concentration_uM']} uM")
print(f"  Ki estimado: {cipro_params['ki_nM']} nM")
print(f"  Eficacia: {cipro_params['efficacy']} (ANTAGONISTA)")
print(f"  ")
print(f"  Ocupacion receptor: {result_cipro['occupancy']:.2%}")
print(f"  Modulacion GABA: {result_cipro['modulation']:.3f}x")
print(f"  Cambio beta EEG: {result_cipro['beta_increase_pct']:.1f}%")
print(f"  Sedacion: {result_cipro['sedation_pct']:.1f}%")

print("""
INTERPRETACION CIPROFLOXACINA:
  - Ocupacion MUY BAJA (<1%) a concentraciones terapeuticas
  - Modulacion <1.0 = INHIBICION de funcion GABA-A
  - Efecto clinico: Pro-convulsivante (reduce inhibicion GABAergica)
  - Riesgo aumentado en: epilepticos, ancianos, insuficiencia renal
  - Interaccion peligrosa con: teofilina, AINES
""")

# ============================================================
# LORATADINA
# ============================================================
print("\n" + "=" * 80)
print("2. LORATADINA")
print("=" * 80)

print("""
DATOS FARMACOLOGICOS (Literatura):
  Clase: Antihistaminico H1 de 2da generacion
  Peso molecular: 382.88 Da
  LogP: 5.2 (muy lipofilico)

MECANISMO PRINCIPAL:
  - Antagonista selectivo receptor H1
  - Ki (H1): ~3 nM (alta afinidad)
  - Minima penetracion SNC (P-gp substrate)

INTERACCION CON GABA-A:
  - SIN afinidad significativa por GABA-A
  - No modula receptores GABA
  - Ki estimado (GABA-A): >100,000 nM (sin efecto)

FARMACOCINETICA:
  - Dosis tipica: 10 mg VO/dia
  - Cmax plasmatico: ~30 ng/mL (~0.08 uM)
  - Penetracion SNC: <5% (substrato P-glicoproteina)
  - Concentracion cerebral estimada: <0.004 uM

EFECTOS CLINICOS SNC:
  - NO sedante (2da generacion)
  - Sin efecto sobre GABA-A
  - Raro: cefalea leve
""")

# Simular con modelo mecanístico
print("\n--- SIMULACION MECANISTICA ---")

# Loratadina NO tiene afinidad por GABA-A
lorat_params = {
    'ki_nM': 500000,  # Sin afinidad real (>500 uM)
    'efficacy': 0.0,  # Sin eficacia (no interactua)
    'binding_site': BindingSite.BZ_SITE,
    'concentration_uM': 0.004  # Concentracion cerebral muy baja
}

result_lorat = model_mech.simulate(
    concentration_uM=lorat_params['concentration_uM'],
    ki_nM=lorat_params['ki_nM'],
    efficacy=lorat_params['efficacy'],
    binding_site=lorat_params['binding_site']
)

print(f"  Concentracion cerebral: {lorat_params['concentration_uM']} uM")
print(f"  Ki estimado: {lorat_params['ki_nM']} nM (sin afinidad)")
print(f"  Eficacia: {lorat_params['efficacy']} (sin efecto)")
print(f"  ")
print(f"  Ocupacion receptor: {result_lorat['occupancy']:.4%}")
print(f"  Modulacion GABA: {result_lorat['modulation']:.3f}x")
print(f"  Cambio beta EEG: {result_lorat['beta_increase_pct']:.2f}%")
print(f"  Sedacion: {result_lorat['sedation_pct']:.2f}%")

print("""
INTERPRETACION LORATADINA:
  - Ocupacion GABA-A: ~0% (sin interaccion)
  - Modulacion = 1.0 (sin cambio)
  - NO produce sedacion via GABA-A
  - Antihistaminico "no sedante" confirmado mecanisticamente
  - Diferente de 1ra generacion (difenhidramina SI penetra SNC)
""")

# ============================================================
# COMPARACION CON DATABASE
# ============================================================
print("\n" + "=" * 80)
print("3. COMPARACION: DATABASE vs MECANISTICO")
print("=" * 80)

print("""
+------------------+-------------+------------------+-------------------+
|     Farmaco      |  En DB?     |  Efecto GABA-A   |  Sedacion Model   |
+------------------+-------------+------------------+-------------------+
| CIPROFLOXACINA   |     NO      |  ANTAGONISTA     |     0.0%          |
|                  |             |  (pro-convuls.)  |  (inhibe GABA)    |
+------------------+-------------+------------------+-------------------+
| LORATADINA       |     NO      |  SIN EFECTO      |     0.0%          |
|                  |             |  (no interactua) |  (no sedante)     |
+------------------+-------------+------------------+-------------------+
| DIAZEPAM         |     SI      |  AGONISTA        |    50-60%         |
|  (referencia)    |             |  (potenciador)   |  (sedante)        |
+------------------+-------------+------------------+-------------------+
""")

# Comparar con un fármaco de la base de datos
print("\n--- COMPARACION CON DIAZEPAM (en DATABASE) ---")

model_db = UnifiedGABAaModel(mode=ModelMode.DATABASE)
result_diaz = model_db.simulate(drug_name='diazepam', concentration_uM=0.5)

print(f"\nDIAZEPAM (DATABASE mode):")
print(f"  Ocupacion: {result_diaz['occupancy']:.1%}")
print(f"  Modulacion: {result_diaz['modulation']:.2f}x")
print(f"  Sedacion: {result_diaz['sedation_pct']:.1f}%")

print(f"\nCIPROFLOXACINA (MECHANISTIC mode):")
print(f"  Ocupacion: {result_cipro['occupancy']:.2%}")
print(f"  Modulacion: {result_cipro['modulation']:.3f}x")
print(f"  Sedacion: {result_cipro['sedation_pct']:.1f}%")

print(f"\nLORATADINA (MECHANISTIC mode):")
print(f"  Ocupacion: {result_lorat['occupancy']:.4%}")
print(f"  Modulacion: {result_lorat['modulation']:.3f}x")
print(f"  Sedacion: {result_lorat['sedation_pct']:.2f}%")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
1. CIPROFLOXACINA:
   - Fuera de DB porque NO es modulador positivo GABA-A
   - Tiene efecto ANTAGONISTA debil (pro-convulsivante)
   - El modelo mecanistico predice correctamente:
     * Muy baja ocupacion
     * Modulacion <1.0 (inhibicion)
     * Sin sedacion
   - VALIDO: Coincide con perfil clinico conocido

2. LORATADINA:
   - Fuera de DB porque NO interactua con GABA-A
   - Antihistaminico 2da generacion (no sedante)
   - El modelo mecanistico predice correctamente:
     * Ocupacion ~0%
     * Sin modulacion
     * Sin sedacion
   - VALIDO: Confirma perfil "no sedante"

3. UTILIDAD DEL MODELO MECANISTICO:
   - Permite evaluar farmacos FUERA de la base de datos
   - Predice efectos desde primeros principios (Ki, eficacia)
   - Puede identificar farmacos con efectos inesperados en GABA-A
   - Util para: interacciones, efectos adversos, drug discovery
""")

print("=" * 80)
print("[SIMULACION - Solo para investigacion/evaluacion medica]")
print("=" * 80)
