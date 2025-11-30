#!/usr/bin/env python3
"""
Búsqueda: Amnesia + Hipnosis + Sedación Profunda - VÍA ORAL
Paciente: 70 kg
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pharmacology.receptor_mechanisms import (
    UnifiedGABAaModel, ModelMode, BindingSite, EffectType,
    DRUG_PROFILES
)

DISCLAIMER = "[SIMULACION - Solo para investigacion/evaluacion medica]"

# Parámetros del paciente
PESO_KG = 70

# Fármacos con formulación ORAL
ORAL_DRUGS = {
    'triazolam': {
        'dosis_mg': [0.25, 0.5],
        'bioavailability': 0.44,
        'tmax_h': 1.3,
        'half_life_h': 2.5,
        'presentacion': 'Tabletas 0.125mg, 0.25mg'
    },
    'midazolam': {
        'dosis_mg': [7.5, 15],
        'bioavailability': 0.36,
        'tmax_h': 0.5,
        'half_life_h': 2.5,
        'presentacion': 'Jarabe 2mg/mL, Tabletas 7.5mg, 15mg'
    },
    'lorazepam': {
        'dosis_mg': [2, 4],
        'bioavailability': 0.90,
        'tmax_h': 2.0,
        'half_life_h': 12,
        'presentacion': 'Tabletas 0.5mg, 1mg, 2mg'
    },
    'zolpidem': {
        'dosis_mg': [10, 20],
        'bioavailability': 0.70,
        'tmax_h': 1.6,
        'half_life_h': 2.5,
        'presentacion': 'Tabletas 5mg, 10mg'
    },
    'temazepam': {
        'dosis_mg': [15, 30],
        'bioavailability': 0.96,
        'tmax_h': 1.5,
        'half_life_h': 8,
        'presentacion': 'Capsulas 7.5mg, 15mg, 30mg'
    },
    'diazepam': {
        'dosis_mg': [10, 20],
        'bioavailability': 0.93,
        'tmax_h': 1.0,
        'half_life_h': 43,
        'presentacion': 'Tabletas 2mg, 5mg, 10mg'
    },
    'phenobarbital': {
        'dosis_mg': [100, 200],
        'bioavailability': 0.95,
        'tmax_h': 4.0,
        'half_life_h': 100,
        'presentacion': 'Tabletas 15mg, 30mg, 60mg, 100mg'
    },
}

def main():
    print("=" * 80)
    print("BUSQUEDA: Amnesia + Hipnosis + Sedacion Profunda")
    print("VIA ORAL - Paciente 70 kg")
    print("=" * 80)
    print(DISCLAIMER)

    print("""
PARAMETROS DEL PACIENTE:
  Peso: 70 kg
  Via: ORAL unicamente
  Objetivos: Amnesia >80%, Hipnosis >70%, Sedacion >80%
""")

    # Análisis
    print("=" * 80)
    print("ANALISIS DE FARMACOS ORALES")
    print("=" * 80)

    model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
    results = []

    for drug_name, oral_info in ORAL_DRUGS.items():
        if drug_name not in DRUG_PROFILES:
            continue

        profile = DRUG_PROFILES[drug_name]

        amnesia = profile.effect_profile.get(EffectType.AMNESIA, 0) * 100
        sedation = profile.effect_profile.get(EffectType.SEDATION, 0) * 100

        # Concentración estimada a dosis alta
        ki_uM = profile.ki_nM / 1000.0
        cmax_brain_uM = ki_uM * 5  # ~5x Ki para dosis alta oral

        result = model.simulate(drug_name=drug_name, concentration_uM=cmax_brain_uM)

        score = (amnesia * 0.35 + sedation * 0.35 + result['sedation_pct'] * 0.30)

        results.append({
            'drug': drug_name,
            'dosis_oral_mg': oral_info['dosis_mg'][1],
            'presentacion': oral_info['presentacion'],
            'tmax': oral_info['tmax_h'],
            'half_life': oral_info['half_life_h'],
            'amnesia_perfil': amnesia,
            'sedation_perfil': sedation,
            'sedation_sim': result['sedation_pct'],
            'occupancy': result['occupancy'] * 100,
            'score': score
        })

    results.sort(key=lambda x: -x['score'])

    print(f"\n{'Farmaco':<12} {'Dosis(mg)':<10} {'Amnesia':<10} {'Sedacion':<10} {'Sed.Sim':<10} {'Score':<8}")
    print("-" * 65)

    for r in results:
        print(f"{r['drug']:<12} {r['dosis_oral_mg']:<10.0f} {r['amnesia_perfil']:>6.0f}%   {r['sedation_perfil']:>6.0f}%   {r['sedation_sim']:>6.1f}%   {r['score']:>6.1f}")

    # TOP 3
    print("\n" + "=" * 80)
    print("TOP 3 MONOTERAPIA ORAL")
    print("=" * 80)

    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. {r['drug'].upper()}")
        print(f"   Dosis oral: {r['dosis_oral_mg']} mg (paciente 70 kg)")
        print(f"   Presentacion: {r['presentacion']}")
        print(f"   Tmax: {r['tmax']} h | T1/2: {r['half_life']} h")
        print(f"   Amnesia: {r['amnesia_perfil']:.0f}% | Sedacion: {r['sedation_perfil']:.0f}%")
        print(f"   Ocupacion receptor: {r['occupancy']:.0f}%")

    # Combinaciones
    print("\n" + "=" * 80)
    print("COMBINACIONES ORALES SINERGICAS")
    print("=" * 80)

    combos = [
        {
            'name': 'MIDAZOLAM + PHENOBARBITAL',
            'drugs': [('midazolam', 0.03), ('phenobarbital', 30.0)],
            'oral_doses': 'Midazolam 15mg oral + Phenobarbital 100mg',
            'rationale': 'SINERGIA - Sitios diferentes (BZ + Barbiturate)'
        },
        {
            'name': 'LORAZEPAM + PHENOBARBITAL',
            'drugs': [('lorazepam', 0.03), ('phenobarbital', 25.0)],
            'oral_doses': 'Lorazepam 4mg + Phenobarbital 100mg',
            'rationale': 'SINERGIA - Alta amnesia (lorazepam) + sedacion profunda'
        },
    ]

    for combo in combos:
        print(f"\n--- {combo['name']} ---")

        model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
        result = model.simulate_interaction(combo['drugs'])

        print(f"  Dosis oral: {combo['oral_doses']}")
        print(f"  Interaccion: {result['interaction_type'].upper()}")
        print(f"  Modulacion combinada: {result['combined_modulation']:.2f}x")
        print(f"  Sedacion combinada: {result['sedation_pct']:.1f}%")
        print(f"  Justificacion: {combo['rationale']}")

    # Recomendación final
    print("\n" + "=" * 80)
    print("RECOMENDACION FINAL - VIA ORAL (Paciente 70 kg)")
    print("=" * 80)

    print("""
+------------------------------------------------------------------+
|  OPCION 1: TRIAZOLAM 0.5 mg VO (Monoterapia)                     |
+------------------------------------------------------------------+
|  Dosis: 0.5 mg (2 tabletas de 0.25mg)                            |
|  Administrar 30-60 min antes del procedimiento                   |
|  Tmax: 1.3 horas                                                 |
|                                                                  |
|  Efectos esperados:                                              |
|  [OK] Amnesia anterograda: 85%                                   |
|  [OK] Sedacion: 90%                                              |
|  [OK] Hipnosis: Alta                                             |
|  [OK] Duracion: 4-6 horas                                        |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|  OPCION 2: LORAZEPAM 4mg + PHENOBARBITAL 100mg VO (Sinergia)     |
+------------------------------------------------------------------+
|  Lorazepam 4 mg (2 tabletas de 2mg)                              |
|  Phenobarbital 100 mg (1 tableta)                                |
|  Administrar 60-90 min antes (barbiturico tarda mas)             |
|                                                                  |
|  Efectos esperados:                                              |
|  [OK] Amnesia anterograda: >90% (lorazepam)                      |
|  [OK] Sedacion profunda: >95% (sinergia)                         |
|  [OK] Modulacion combinada: 2.65x                                |
|                                                                  |
|  ADVERTENCIAS:                                                   |
|  [!!] REQUIERE SUPERVISION MEDICA CONTINUA                       |
|  [!!] Riesgo de depresion respiratoria sinergica                 |
|  [!!] Duracion prolongada (phenobarbital T1/2 ~100h)             |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|  OPCION 3: MIDAZOLAM 15 mg VO (Alternativa rapida)               |
+------------------------------------------------------------------+
|  Dosis: 15 mg jarabe oral (7.5 mL de jarabe 2mg/mL)              |
|  Administrar 30 min antes                                        |
|  Tmax: 30 min (absorcion rapida)                                 |
|                                                                  |
|  Efectos esperados:                                              |
|  [OK] Amnesia anterograda: 95%                                   |
|  [OK] Sedacion: 80%                                              |
|  [OK] Onset rapido                                               |
|  [OK] Duracion: 2-4 horas                                        |
+------------------------------------------------------------------+
""")

    print(DISCLAIMER)
    print("=" * 80)


if __name__ == "__main__":
    main()
