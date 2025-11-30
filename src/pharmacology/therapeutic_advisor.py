#!/usr/bin/env python3
"""
Módulo de Asesoría Terapéutica Inversa
======================================

Sistema de sugerencia farmacológica basado en parámetros clínicos del paciente.
Utiliza la base de datos de fármacos y el modelo mecanístico calibrado para
generar recomendaciones personalizadas.

⚠️ DISCLAIMER IMPORTANTE ⚠️
Este sistema es EXCLUSIVAMENTE para fines de:
- Investigación científica
- Evaluación por colegios médicos
- Docencia y formación académica
- Desarrollo y validación de modelos computacionales

NO ESTÁ DISEÑADO NI APROBADO PARA:
- Diagnóstico clínico real
- Prescripción médica a pacientes
- Toma de decisiones terapéuticas sin supervisión médica
- Sustitución del juicio clínico profesional

Las sugerencias generadas son SIMULACIONES TEÓRICAS basadas en modelos
matemáticos y NO constituyen consejo médico.

Autor: Francisco Molina Burgos (Yatrogenesis)
Versión: 1.0 - Para evaluación por colegios médicos
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# Importar módulos del sistema
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.pharmacology.receptor_mechanisms import (
    UnifiedGABAaModel, ModelMode, BindingSite, EffectType,
    DRUG_PROFILES, DrugMolecularProfile
)


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

DISCLAIMER_FULL = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  AVISO LEGAL Y MÉDICO IMPORTANTE  ⚠️                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Este sistema de asesoría terapéutica es una herramienta de SIMULACIÓN       ║
║  diseñada EXCLUSIVAMENTE para:                                               ║
║                                                                              ║
║    ✓ Investigación científica y académica                                    ║
║    ✓ Evaluación por colegios médicos y comités de ética                      ║
║    ✓ Docencia en farmacología computacional                                  ║
║    ✓ Validación de modelos predictivos                                       ║
║                                                                              ║
║  ⛔ NO ESTÁ APROBADO para uso clínico real                                   ║
║  ⛔ NO sustituye el criterio médico profesional                              ║
║  ⛔ NO debe usarse para prescripción sin supervisión                         ║
║                                                                              ║
║  Las predicciones se basan en modelos matemáticos (Black & Leff, 1983)       ║
║  y datos farmacocinéticos de literatura. La respuesta individual puede       ║
║  variar significativamente debido a:                                         ║
║                                                                              ║
║    • Polimorfismos genéticos (CYP450, receptores)                           ║
║    • Comorbilidades no consideradas                                          ║
║    • Interacciones farmacológicas adicionales                                ║
║    • Factores ambientales y dietéticos                                       ║
║                                                                              ║
║  Cualquier decisión terapéutica debe ser tomada por un profesional           ║
║  médico calificado tras evaluación completa del paciente.                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

DISCLAIMER_SHORT = "[⚠️ SIMULACIÓN - Solo para investigación/evaluación médica]"


class ClinicalGoal(Enum):
    """Objetivos terapéuticos disponibles."""
    ANXIOLYSIS = "anxiolysis"
    SEDATION = "sedation"
    HYPNOSIS = "hypnosis"
    ANESTHESIA = "anesthesia"
    ANTICONVULSANT = "anticonvulsant"
    MUSCLE_RELAXATION = "muscle_relaxation"
    AMNESIA = "amnesia"


class PatientProfile(Enum):
    """Perfiles de paciente para ajuste de dosis."""
    STANDARD = "standard"
    ELDERLY = "elderly"
    PEDIATRIC = "pediatric"
    HEPATIC_IMPAIRMENT = "hepatic_impairment"
    RENAL_IMPAIRMENT = "renal_impairment"
    OBESE = "obese"
    UNDERWEIGHT = "underweight"


class ContraindicationType(Enum):
    """Tipos de contraindicaciones."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    CAUTION = "caution"


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class PatientParameters:
    """Parámetros clínicos del paciente para evaluación."""
    # Datos demográficos
    age: int = 40
    weight_kg: float = 70.0
    sex: str = "M"

    # Objetivos terapéuticos (0-100%)
    target_anxiolysis: float = 0.0
    target_sedation: float = 0.0
    target_amnesia: float = 0.0
    target_anticonvulsant: float = 0.0
    target_anesthesia: float = 0.0

    # Perfil del paciente
    profile: PatientProfile = PatientProfile.STANDARD

    # Contraindicaciones (lista de fármacos)
    contraindicated_drugs: List[str] = field(default_factory=list)
    contraindicated_classes: List[str] = field(default_factory=list)

    # Medicación actual (para interacciones)
    current_medications: List[str] = field(default_factory=list)

    # Historial relevante
    history_respiratory_depression: bool = False
    history_paradoxical_reaction: bool = False
    history_substance_abuse: bool = False
    pregnancy_risk: bool = False

    # Preferencias
    prefer_short_acting: bool = False
    prefer_reversible: bool = False
    avoid_amnesia: bool = False


@dataclass
class DrugSuggestion:
    """Sugerencia de fármaco individual."""
    drug_name: str
    binding_site: str
    suggested_concentration_uM: float

    # Efectos predichos
    predicted_sedation: float
    predicted_anxiolysis: float
    predicted_amnesia: float
    predicted_anticonvulsant: float
    predicted_anesthesia: float

    # Métricas de ajuste
    target_match_score: float  # 0-100%
    safety_score: float  # 0-100%
    overall_score: float  # 0-100%

    # Justificación
    rationale: str
    warnings: List[str] = field(default_factory=list)
    dose_adjustment: str = "standard"


@dataclass
class CombinationSuggestion:
    """Sugerencia de combinación farmacológica."""
    drugs: List[Tuple[str, float]]  # (nombre, concentración)
    interaction_type: str  # synergy, additive, competition

    # Efectos predichos combinados
    combined_sedation: float
    combined_anxiolysis: float
    combined_modulation: float

    # Métricas
    synergy_benefit: float  # Beneficio de la sinergia
    target_match_score: float
    safety_score: float
    overall_score: float

    # Justificación
    rationale: str
    clinical_precedent: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class TherapeuticReport:
    """Reporte completo de sugerencias terapéuticas."""
    timestamp: str
    patient_parameters: Dict

    # Sugerencias
    monotherapy_suggestions: List[DrugSuggestion]
    combination_suggestions: List[CombinationSuggestion]

    # Metadatos
    model_version: str = "1.0"
    disclaimer_accepted: bool = False
    evaluation_purpose: str = ""


# =============================================================================
# SISTEMA DE CONTRAINDICACIONES
# =============================================================================

CONTRAINDICATIONS_DATABASE = {
    # Contraindicaciones absolutas
    "respiratory_depression": {
        "drugs": ["propofol", "thiopental", "midazolam"],
        "type": ContraindicationType.ABSOLUTE,
        "reason": "Riesgo de depresión respiratoria severa"
    },
    "myasthenia_gravis": {
        "drugs": ["diazepam", "lorazepam", "clonazepam"],
        "type": ContraindicationType.RELATIVE,
        "reason": "Pueden exacerbar debilidad muscular"
    },
    "acute_angle_glaucoma": {
        "drugs": ["diazepam", "alprazolam", "lorazepam"],
        "type": ContraindicationType.RELATIVE,
        "reason": "Efecto anticolinérgico puede elevar presión intraocular"
    },
    "pregnancy_first_trimester": {
        "classes": ["bz_site", "barbiturate"],
        "type": ContraindicationType.ABSOLUTE,
        "reason": "Riesgo teratogénico documentado"
    },
    "substance_abuse_history": {
        "drugs": ["alprazolam", "triazolam", "zolpidem"],
        "type": ContraindicationType.RELATIVE,
        "reason": "Alto potencial de abuso - preferir alternativas"
    },
    "elderly_fall_risk": {
        "drugs": ["triazolam", "zolpidem", "zaleplon"],
        "type": ContraindicationType.CAUTION,
        "reason": "Riesgo aumentado de caídas nocturnas"
    },
    "hepatic_impairment_severe": {
        "drugs": ["diazepam", "alprazolam", "midazolam"],
        "type": ContraindicationType.RELATIVE,
        "reason": "Metabolismo hepático reducido - acumulación"
    },
}


# Perfiles de ajuste de dosis
DOSE_ADJUSTMENTS = {
    PatientProfile.ELDERLY: {
        "factor": 0.5,
        "reason": "Reducir 50% - Sensibilidad aumentada, clearance reducido"
    },
    PatientProfile.PEDIATRIC: {
        "factor": 0.7,
        "reason": "Ajustar por peso y metabolismo diferente"
    },
    PatientProfile.HEPATIC_IMPAIRMENT: {
        "factor": 0.5,
        "reason": "Reducir 50% - Metabolismo hepático comprometido"
    },
    PatientProfile.RENAL_IMPAIRMENT: {
        "factor": 0.75,
        "reason": "Reducir 25% - Eliminación renal afectada"
    },
    PatientProfile.OBESE: {
        "factor": 1.0,  # Usar peso ideal, no real
        "reason": "Calcular con peso ideal ajustado"
    },
    PatientProfile.STANDARD: {
        "factor": 1.0,
        "reason": "Dosis estándar"
    },
}


# =============================================================================
# CLASE PRINCIPAL: ASESOR TERAPÉUTICO
# =============================================================================

class TherapeuticAdvisor:
    """
    Sistema de asesoría terapéutica inversa.

    Genera sugerencias de fármacos basadas en objetivos clínicos
    del paciente, utilizando el modelo mecanístico calibrado.
    """

    def __init__(self, show_disclaimer: bool = True):
        """Inicializa el asesor terapéutico."""
        self.model = UnifiedGABAaModel(mode=ModelMode.DATABASE)
        self.drug_profiles = DRUG_PROFILES

        if show_disclaimer:
            print(DISCLAIMER_FULL)

    def analyze_patient(self, params: PatientParameters) -> TherapeuticReport:
        """
        Analiza parámetros del paciente y genera sugerencias.

        Args:
            params: Parámetros clínicos del paciente

        Returns:
            TherapeuticReport con sugerencias ordenadas por relevancia
        """
        # 1. Filtrar fármacos contraindicados
        available_drugs = self._filter_contraindicated(params)

        # 2. Evaluar cada fármaco disponible
        monotherapy = self._evaluate_monotherapy(params, available_drugs)

        # 3. Evaluar combinaciones potenciales
        combinations = self._evaluate_combinations(params, available_drugs)

        # 4. Generar reporte
        report = TherapeuticReport(
            timestamp=datetime.now().isoformat(),
            patient_parameters=self._params_to_dict(params),
            monotherapy_suggestions=sorted(monotherapy, key=lambda x: -x.overall_score)[:5],
            combination_suggestions=sorted(combinations, key=lambda x: -x.overall_score)[:3],
        )

        return report

    def _filter_contraindicated(self, params: PatientParameters) -> List[str]:
        """Filtra fármacos contraindicados para el paciente."""
        available = list(self.drug_profiles.keys())
        excluded = set()

        # Exclusiones directas del paciente
        excluded.update(params.contraindicated_drugs)

        # Exclusiones por clase
        for drug, profile in self.drug_profiles.items():
            if profile.binding_site.value in params.contraindicated_classes:
                excluded.add(drug)

        # Exclusiones por historial
        if params.history_respiratory_depression:
            excluded.update(CONTRAINDICATIONS_DATABASE["respiratory_depression"]["drugs"])

        if params.history_substance_abuse:
            excluded.update(CONTRAINDICATIONS_DATABASE["substance_abuse_history"]["drugs"])

        if params.pregnancy_risk:
            for drug, profile in self.drug_profiles.items():
                if profile.binding_site.value in ["bz_site", "barbiturate"]:
                    excluded.add(drug)

        # Exclusiones por perfil
        if params.profile == PatientProfile.ELDERLY:
            excluded.update(CONTRAINDICATIONS_DATABASE["elderly_fall_risk"]["drugs"])

        if params.profile == PatientProfile.HEPATIC_IMPAIRMENT:
            excluded.update(CONTRAINDICATIONS_DATABASE["hepatic_impairment_severe"]["drugs"])

        # Preferencias
        if params.avoid_amnesia:
            # Excluir fármacos con alto perfil amnésico
            for drug, profile in self.drug_profiles.items():
                if profile.effect_profile.get(EffectType.AMNESIA, 0) > 0.7:
                    excluded.add(drug)

        return [d for d in available if d not in excluded]

    def _evaluate_monotherapy(
        self,
        params: PatientParameters,
        available_drugs: List[str]
    ) -> List[DrugSuggestion]:
        """Evalúa fármacos individuales para el paciente."""
        suggestions = []

        for drug_name in available_drugs:
            profile = self.drug_profiles[drug_name]

            # Encontrar concentración óptima para objetivos
            optimal_conc, effects = self._find_optimal_concentration(
                drug_name, params
            )

            # Calcular scores
            target_match = self._calculate_target_match(effects, params)
            safety_score = self._calculate_safety_score(drug_name, optimal_conc, params)

            # Ajuste de dosis
            dose_adj = DOSE_ADJUSTMENTS[params.profile]
            adjusted_conc = optimal_conc * dose_adj["factor"]

            # Generar warnings
            warnings = self._generate_warnings(drug_name, params)

            # Generar rationale
            rationale = self._generate_rationale(drug_name, effects, params)

            suggestion = DrugSuggestion(
                drug_name=drug_name,
                binding_site=profile.binding_site.value,
                suggested_concentration_uM=adjusted_conc,
                predicted_sedation=effects["sedation"],
                predicted_anxiolysis=effects["anxiolysis"],
                predicted_amnesia=effects["amnesia"],
                predicted_anticonvulsant=effects["anticonvulsant"],
                predicted_anesthesia=effects["anesthesia"],
                target_match_score=target_match,
                safety_score=safety_score,
                overall_score=(target_match * 0.6 + safety_score * 0.4),
                rationale=rationale,
                warnings=warnings,
                dose_adjustment=dose_adj["reason"]
            )

            suggestions.append(suggestion)

        return suggestions

    def _find_optimal_concentration(
        self,
        drug_name: str,
        params: PatientParameters
    ) -> Tuple[float, Dict]:
        """Encuentra la concentración óptima para los objetivos."""
        profile = self.drug_profiles[drug_name]
        ki_uM = profile.ki_nM / 1000.0

        # Probar rango de concentraciones
        best_conc = ki_uM
        best_score = -1
        best_effects = {}

        for mult in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
            conc = ki_uM * mult

            # Simular
            result = self.model.simulate(drug_name=drug_name, concentration_uM=conc)

            # Extraer efectos
            effects = {
                "sedation": result["sedation_pct"],
                "anxiolysis": profile.effect_profile.get(EffectType.ANXIOLYSIS, 0) * 100 * result["occupancy"],
                "amnesia": profile.effect_profile.get(EffectType.AMNESIA, 0) * 100 * result["occupancy"],
                "anticonvulsant": profile.effect_profile.get(EffectType.ANTICONVULSANT, 0) * 100 * result["occupancy"],
                "anesthesia": profile.effect_profile.get(EffectType.ANESTHESIA, 0) * 100 * result["occupancy"],
            }

            # Calcular score vs objetivos
            score = self._calculate_target_match(effects, params)

            if score > best_score:
                best_score = score
                best_conc = conc
                best_effects = effects

        return best_conc, best_effects

    def _calculate_target_match(self, effects: Dict, params: PatientParameters) -> float:
        """Calcula qué tan bien coinciden los efectos con los objetivos."""
        targets = {
            "sedation": params.target_sedation,
            "anxiolysis": params.target_anxiolysis,
            "amnesia": params.target_amnesia if not params.avoid_amnesia else 0,
            "anticonvulsant": params.target_anticonvulsant,
            "anesthesia": params.target_anesthesia,
        }

        # Calcular diferencia ponderada
        total_weight = 0
        weighted_match = 0

        for effect, target in targets.items():
            if target > 0:
                actual = effects.get(effect, 0)
                # Penalizar más si excede (sobredosis) que si queda corto
                if actual > target:
                    diff = (actual - target) / 100 * 1.5  # Penalización extra
                else:
                    diff = (target - actual) / 100

                match = max(0, 1 - diff)
                weighted_match += match * target
                total_weight += target

        if total_weight == 0:
            return 50.0  # Sin objetivos específicos

        return (weighted_match / total_weight) * 100

    def _calculate_safety_score(
        self,
        drug_name: str,
        concentration: float,
        params: PatientParameters
    ) -> float:
        """Calcula score de seguridad basado en múltiples factores."""
        score = 100.0
        profile = self.drug_profiles[drug_name]
        ki_uM = profile.ki_nM / 1000.0

        # Penalizar concentraciones muy altas (>10x Ki)
        if concentration > ki_uM * 10:
            score -= 20

        # Penalizar alto potencial amnésico si hay preferencia
        if params.avoid_amnesia:
            amnesia_risk = profile.effect_profile.get(EffectType.AMNESIA, 0)
            score -= amnesia_risk * 30

        # Penalizar en perfiles de riesgo
        if params.profile == PatientProfile.ELDERLY:
            # Penalizar fármacos de larga duración
            if drug_name in ["diazepam", "clonazepam", "phenobarbital"]:
                score -= 15

        if params.profile == PatientProfile.HEPATIC_IMPAIRMENT:
            # Penalizar metabolismo hepático intensivo
            if drug_name in ["diazepam", "midazolam", "alprazolam"]:
                score -= 20

        # Bonus por reversibilidad si se prefiere
        if params.prefer_reversible:
            if drug_name == "midazolam":  # Flumazenil disponible
                score += 10

        # Bonus por acción corta si se prefiere
        if params.prefer_short_acting:
            if drug_name in ["triazolam", "zaleplon", "etomidate"]:
                score += 10

        return max(0, min(100, score))

    def _generate_warnings(self, drug_name: str, params: PatientParameters) -> List[str]:
        """Genera advertencias específicas para el fármaco y paciente."""
        warnings = []
        profile = self.drug_profiles[drug_name]

        # Advertencias generales
        warnings.append(DISCLAIMER_SHORT)

        # Por perfil de paciente
        if params.profile == PatientProfile.ELDERLY:
            warnings.append("⚠️ Paciente geriátrico: iniciar con dosis mínima, titular lentamente")

        if params.profile == PatientProfile.HEPATIC_IMPAIRMENT:
            warnings.append("⚠️ Insuficiencia hepática: monitorizar función hepática")

        # Por fármaco específico
        if profile.binding_site == BindingSite.ANESTHETIC_SITE:
            warnings.append("⚠️ Requiere monitorización de vía aérea y signos vitales")

        if drug_name in ["propofol", "thiopental"]:
            warnings.append("⚠️ Administración IV requiere acceso venoso y equipo de reanimación")

        if profile.effect_profile.get(EffectType.AMNESIA, 0) > 0.7:
            warnings.append("⚠️ Alto potencial amnésico - informar al paciente")

        # Interacciones con medicación actual
        for current_med in params.current_medications:
            if current_med.lower() in ["opioides", "opioid", "morfina", "fentanilo"]:
                warnings.append("⚠️ INTERACCIÓN: Riesgo de depresión respiratoria con opioides")
            if current_med.lower() in ["alcohol", "etanol"]:
                warnings.append("⚠️ INTERACCIÓN: Potenciación severa con alcohol - contraindicado")

        return warnings

    def _generate_rationale(
        self,
        drug_name: str,
        effects: Dict,
        params: PatientParameters
    ) -> str:
        """Genera justificación clínica para la sugerencia."""
        profile = self.drug_profiles[drug_name]

        rationale_parts = []

        # Mecanismo
        site_names = {
            "bz_site": "sitio benzodiazepínico (interfaz α-γ)",
            "anesthetic": "sitio anestésico (subunidad β TM2-TM3)",
            "barbiturate": "sitio barbitúrico",
            "neurosteroid": "sitio neuroesteroide"
        }
        site_name = site_names.get(profile.binding_site.value, profile.binding_site.value)
        rationale_parts.append(f"Modulador alostérico positivo del receptor GABA-A en {site_name}.")

        # Perfil de efectos
        effect_desc = []
        if effects["anxiolysis"] > 50:
            effect_desc.append(f"ansiolisis ({effects['anxiolysis']:.0f}%)")
        if effects["sedation"] > 50:
            effect_desc.append(f"sedación ({effects['sedation']:.0f}%)")
        if effects["anticonvulsant"] > 50:
            effect_desc.append(f"anticonvulsivante ({effects['anticonvulsant']:.0f}%)")

        if effect_desc:
            rationale_parts.append(f"Proporciona: {', '.join(effect_desc)}.")

        # Ventajas específicas
        if drug_name == "midazolam" and params.prefer_reversible:
            rationale_parts.append("Ventaja: reversible con flumazenil.")

        if drug_name == "lorazepam" and params.profile == PatientProfile.HEPATIC_IMPAIRMENT:
            rationale_parts.append("Ventaja: glucuronidación directa, menor dependencia hepática.")

        return " ".join(rationale_parts)

    def _evaluate_combinations(
        self,
        params: PatientParameters,
        available_drugs: List[str]
    ) -> List[CombinationSuggestion]:
        """Evalúa combinaciones farmacológicas potencialmente sinérgicas."""
        combinations = []

        # Identificar fármacos por sitio de unión
        by_site = {}
        for drug in available_drugs:
            site = self.drug_profiles[drug].binding_site.value
            if site not in by_site:
                by_site[site] = []
            by_site[site].append(drug)

        # Solo considerar combinaciones de diferentes sitios (sinergia)
        sites = list(by_site.keys())

        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                # Tomar el mejor de cada sitio
                drugs1 = by_site[site1]
                drugs2 = by_site[site2]

                # Evaluar combinaciones limitadas
                for drug1 in drugs1[:2]:  # Top 2 de cada sitio
                    for drug2 in drugs2[:2]:
                        combo = self._evaluate_single_combination(
                            drug1, drug2, params
                        )
                        if combo:
                            combinations.append(combo)

        return combinations

    def _evaluate_single_combination(
        self,
        drug1: str,
        drug2: str,
        params: PatientParameters
    ) -> Optional[CombinationSuggestion]:
        """Evalúa una combinación específica de dos fármacos."""
        profile1 = self.drug_profiles[drug1]
        profile2 = self.drug_profiles[drug2]

        # Determinar tipo de interacción
        if profile1.binding_site == profile2.binding_site:
            interaction_type = "competition"
            synergy_benefit = 0
        else:
            interaction_type = "synergy"
            synergy_benefit = 25  # Beneficio base de sinergia

        # Calcular concentraciones reducidas para combinación
        ki1 = profile1.ki_nM / 1000.0
        ki2 = profile2.ki_nM / 1000.0

        # Usar dosis subóptimas individuales
        conc1 = ki1 * 2  # 2x Ki (menor que monoterapia óptima)
        conc2 = ki2 * 2

        # Simular combinación
        self.model = UnifiedGABAaModel(mode=ModelMode.DATABASE)  # Reset
        result = self.model.simulate_interaction([(drug1, conc1), (drug2, conc2)])

        # Calcular efectos combinados
        combined_sedation = result["sedation_pct"]
        combined_mod = result["combined_modulation"]

        # Estimar ansiolisis combinada
        anx1 = profile1.effect_profile.get(EffectType.ANXIOLYSIS, 0)
        anx2 = profile2.effect_profile.get(EffectType.ANXIOLYSIS, 0)
        combined_anxiolysis = min(100, (anx1 + anx2) * 50 * (combined_mod / 2))

        # Calcular scores
        effects = {
            "sedation": combined_sedation,
            "anxiolysis": combined_anxiolysis,
            "amnesia": 50,  # Estimado
            "anticonvulsant": 50,
            "anesthesia": combined_sedation * 0.8,
        }

        target_match = self._calculate_target_match(effects, params)

        # Safety score para combinación (más conservador)
        safety1 = self._calculate_safety_score(drug1, conc1, params)
        safety2 = self._calculate_safety_score(drug2, conc2, params)
        safety_score = min(safety1, safety2) * 0.8  # Penalización por combinación

        # Warnings específicos de combinación
        warnings = [
            DISCLAIMER_SHORT,
            "⚠️ Combinación farmacológica: requiere monitorización intensiva",
            f"⚠️ Interacción {interaction_type.upper()} - ajustar dosis individuales"
        ]

        if combined_sedation > 80:
            warnings.append("⚠️ Alto nivel de sedación combinada - riesgo de depresión respiratoria")

        # Precedente clínico
        clinical_precedent = self._get_clinical_precedent(drug1, drug2)

        return CombinationSuggestion(
            drugs=[(drug1, conc1), (drug2, conc2)],
            interaction_type=interaction_type,
            combined_sedation=combined_sedation,
            combined_anxiolysis=combined_anxiolysis,
            combined_modulation=combined_mod,
            synergy_benefit=synergy_benefit,
            target_match_score=target_match,
            safety_score=safety_score,
            overall_score=(target_match * 0.5 + safety_score * 0.3 + synergy_benefit * 0.2),
            rationale=f"Combinación de {drug1} ({profile1.binding_site.value}) + {drug2} ({profile2.binding_site.value}) "
                     f"para efecto {interaction_type}. Modulación combinada: {combined_mod:.2f}x",
            clinical_precedent=clinical_precedent,
            warnings=warnings
        )

    def _get_clinical_precedent(self, drug1: str, drug2: str) -> str:
        """Retorna precedente clínico conocido para la combinación."""
        precedents = {
            ("midazolam", "propofol"): "Combinación establecida en anestesia - permite reducir dosis de ambos (co-inducción)",
            ("midazolam", "ketamine"): "Combinación para sedación procedural - el BZ reduce efectos psicotomiméticos del ketamine",
            ("diazepam", "propofol"): "Premedicación con BZ reduce requerimientos de propofol",
            ("lorazepam", "phenobarbital"): "Combinación anticonvulsivante de segunda línea en status epilepticus",
        }

        key = tuple(sorted([drug1, drug2]))
        return precedents.get(key, "Combinación basada en mecanismo - verificar literatura específica")

    def _params_to_dict(self, params: PatientParameters) -> Dict:
        """Convierte parámetros a diccionario para el reporte."""
        return {
            "age": params.age,
            "weight_kg": params.weight_kg,
            "sex": params.sex,
            "profile": params.profile.value,
            "targets": {
                "anxiolysis": params.target_anxiolysis,
                "sedation": params.target_sedation,
                "amnesia": params.target_amnesia,
                "anticonvulsant": params.target_anticonvulsant,
                "anesthesia": params.target_anesthesia,
            },
            "contraindications": params.contraindicated_drugs,
            "current_medications": params.current_medications,
            "preferences": {
                "short_acting": params.prefer_short_acting,
                "reversible": params.prefer_reversible,
                "avoid_amnesia": params.avoid_amnesia,
            }
        }

    def print_report(self, report: TherapeuticReport):
        """Imprime el reporte de forma legible."""
        print("\n" + "=" * 80)
        print("REPORTE DE ASESORÍA TERAPÉUTICA")
        print("=" * 80)
        print(DISCLAIMER_SHORT)
        print(f"\nFecha: {report.timestamp}")

        # Parámetros del paciente
        print("\n--- PARÁMETROS DEL PACIENTE ---")
        p = report.patient_parameters
        print(f"  Edad: {p['age']} años | Peso: {p['weight_kg']} kg | Sexo: {p['sex']}")
        print(f"  Perfil: {p['profile']}")
        print(f"  Objetivos:")
        for target, value in p['targets'].items():
            if value > 0:
                print(f"    - {target}: {value}%")

        if p['contraindications']:
            print(f"  Contraindicaciones: {', '.join(p['contraindications'])}")
        if p['current_medications']:
            print(f"  Medicación actual: {', '.join(p['current_medications'])}")

        # Sugerencias de monoterapia
        print("\n--- SUGERENCIAS DE MONOTERAPIA ---")
        for i, sug in enumerate(report.monotherapy_suggestions, 1):
            print(f"\n  {i}. {sug.drug_name.upper()}")
            print(f"     Sitio: {sug.binding_site}")
            print(f"     Concentración sugerida: {sug.suggested_concentration_uM:.3f} µM")
            print(f"     Ajuste: {sug.dose_adjustment}")
            print(f"     Efectos predichos:")
            print(f"       - Sedación: {sug.predicted_sedation:.1f}%")
            print(f"       - Ansiolisis: {sug.predicted_anxiolysis:.1f}%")
            print(f"       - Amnesia: {sug.predicted_amnesia:.1f}%")
            print(f"     Scores: Match={sug.target_match_score:.1f}% | Safety={sug.safety_score:.1f}% | Overall={sug.overall_score:.1f}%")
            print(f"     Justificación: {sug.rationale}")
            for w in sug.warnings:
                print(f"     {w}")

        # Sugerencias de combinación
        if report.combination_suggestions:
            print("\n--- SUGERENCIAS DE COMBINACIÓN ---")
            for i, combo in enumerate(report.combination_suggestions, 1):
                drugs_str = " + ".join([f"{d[0]} ({d[1]:.3f} µM)" for d in combo.drugs])
                print(f"\n  {i}. {drugs_str}")
                print(f"     Tipo de interacción: {combo.interaction_type.upper()}")
                print(f"     Modulación combinada: {combo.combined_modulation:.2f}x")
                print(f"     Sedación combinada: {combo.combined_sedation:.1f}%")
                print(f"     Ansiolisis combinada: {combo.combined_anxiolysis:.1f}%")
                print(f"     Scores: Match={combo.target_match_score:.1f}% | Safety={combo.safety_score:.1f}% | Overall={combo.overall_score:.1f}%")
                print(f"     Precedente clínico: {combo.clinical_precedent}")
                print(f"     Justificación: {combo.rationale}")
                for w in combo.warnings:
                    print(f"     {w}")

        print("\n" + "=" * 80)
        print(DISCLAIMER_SHORT)
        print("=" * 80)

    def export_report(self, report: TherapeuticReport, filepath: str):
        """Exporta el reporte a JSON."""
        # Convertir a diccionario serializable
        data = {
            "disclaimer": DISCLAIMER_FULL,
            "timestamp": report.timestamp,
            "model_version": report.model_version,
            "patient_parameters": report.patient_parameters,
            "monotherapy_suggestions": [
                {
                    "drug": s.drug_name,
                    "binding_site": s.binding_site,
                    "concentration_uM": s.suggested_concentration_uM,
                    "effects": {
                        "sedation": s.predicted_sedation,
                        "anxiolysis": s.predicted_anxiolysis,
                        "amnesia": s.predicted_amnesia,
                        "anticonvulsant": s.predicted_anticonvulsant,
                        "anesthesia": s.predicted_anesthesia,
                    },
                    "scores": {
                        "target_match": s.target_match_score,
                        "safety": s.safety_score,
                        "overall": s.overall_score,
                    },
                    "rationale": s.rationale,
                    "dose_adjustment": s.dose_adjustment,
                    "warnings": s.warnings,
                }
                for s in report.monotherapy_suggestions
            ],
            "combination_suggestions": [
                {
                    "drugs": c.drugs,
                    "interaction_type": c.interaction_type,
                    "combined_effects": {
                        "modulation": c.combined_modulation,
                        "sedation": c.combined_sedation,
                        "anxiolysis": c.combined_anxiolysis,
                    },
                    "scores": {
                        "target_match": c.target_match_score,
                        "safety": c.safety_score,
                        "overall": c.overall_score,
                        "synergy_benefit": c.synergy_benefit,
                    },
                    "rationale": c.rationale,
                    "clinical_precedent": c.clinical_precedent,
                    "warnings": c.warnings,
                }
                for c in report.combination_suggestions
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nReporte exportado a: {filepath}")


# =============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# =============================================================================

def interactive_consultation():
    """Interfaz interactiva para consulta terapéutica."""
    print("\n" + "#" * 80)
    print("#" + " " * 20 + "ASESOR TERAPÉUTICO INVERSO" + " " * 24 + "#")
    print("#" + " " * 15 + "Sistema de Evaluación para Colegios Médicos" + " " * 14 + "#")
    print("#" * 80)

    advisor = TherapeuticAdvisor(show_disclaimer=True)

    print("\n¿Acepta el disclaimer y confirma uso exclusivo para evaluación/investigación?")
    print("(Este sistema NO es para uso clínico real)")
    response = input("\nEscriba 'ACEPTO' para continuar: ")

    if response.strip().upper() != "ACEPTO":
        print("Consulta cancelada.")
        return

    print("\n--- INGRESO DE PARÁMETROS DEL PACIENTE ---")

    # Recopilar datos
    try:
        age = int(input("Edad del paciente (años): ") or "40")
        weight = float(input("Peso (kg): ") or "70")
        sex = input("Sexo (M/F): ") or "M"

        print("\n--- OBJETIVOS TERAPÉUTICOS (0-100%) ---")
        target_anx = float(input("Objetivo ansiolisis (%): ") or "0")
        target_sed = float(input("Objetivo sedación (%): ") or "0")
        target_amnesia = float(input("Objetivo amnesia (%): ") or "0")
        target_anticonv = float(input("Objetivo anticonvulsivante (%): ") or "0")
        target_anest = float(input("Objetivo anestesia (%): ") or "0")

        print("\n--- PERFIL DEL PACIENTE ---")
        print("1. Estándar")
        print("2. Geriátrico")
        print("3. Pediátrico")
        print("4. Insuficiencia hepática")
        print("5. Insuficiencia renal")
        profile_choice = input("Seleccione perfil (1-5): ") or "1"

        profiles = {
            "1": PatientProfile.STANDARD,
            "2": PatientProfile.ELDERLY,
            "3": PatientProfile.PEDIATRIC,
            "4": PatientProfile.HEPATIC_IMPAIRMENT,
            "5": PatientProfile.RENAL_IMPAIRMENT,
        }
        profile = profiles.get(profile_choice, PatientProfile.STANDARD)

        # Contraindicaciones
        contra = input("\nFármacos contraindicados (separados por coma, o Enter para ninguno): ")
        contraindicated = [c.strip().lower() for c in contra.split(",") if c.strip()]

        # Medicación actual
        current = input("Medicación actual (separados por coma, o Enter para ninguno): ")
        current_meds = [c.strip() for c in current.split(",") if c.strip()]

        # Preferencias
        print("\n--- PREFERENCIAS ---")
        avoid_amnesia = input("¿Evitar fármacos con alto potencial amnésico? (s/n): ").lower() == "s"
        prefer_short = input("¿Preferir acción corta? (s/n): ").lower() == "s"
        prefer_reversible = input("¿Preferir reversible con antagonista? (s/n): ").lower() == "s"

        # Crear parámetros
        params = PatientParameters(
            age=age,
            weight_kg=weight,
            sex=sex,
            target_anxiolysis=target_anx,
            target_sedation=target_sed,
            target_amnesia=target_amnesia,
            target_anticonvulsant=target_anticonv,
            target_anesthesia=target_anest,
            profile=profile,
            contraindicated_drugs=contraindicated,
            current_medications=current_meds,
            avoid_amnesia=avoid_amnesia,
            prefer_short_acting=prefer_short,
            prefer_reversible=prefer_reversible,
        )

        # Generar reporte
        print("\n--- GENERANDO ANÁLISIS ---")
        report = advisor.analyze_patient(params)

        # Mostrar resultados
        advisor.print_report(report)

        # Exportar
        export = input("\n¿Exportar reporte a JSON? (s/n): ").lower()
        if export == "s":
            filepath = Path(__file__).parent.parent.parent / "results" / f"therapeutic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath.parent.mkdir(exist_ok=True)
            advisor.export_report(report, str(filepath))

    except ValueError as e:
        print(f"Error en entrada: {e}")
    except KeyboardInterrupt:
        print("\nConsulta cancelada.")


def demo_cases():
    """Ejecuta casos de demostración para evaluación."""
    print("\n" + "#" * 80)
    print("#" + " " * 20 + "CASOS DE DEMOSTRACIÓN" + " " * 29 + "#")
    print("#" + " " * 15 + "Para Evaluación por Colegios Médicos" + " " * 19 + "#")
    print("#" * 80)

    advisor = TherapeuticAdvisor(show_disclaimer=True)

    # Caso 1: Ansiedad aguda
    print("\n" + "=" * 80)
    print("CASO 1: Ansiedad aguda pre-procedimiento")
    print("=" * 80)

    params1 = PatientParameters(
        age=45,
        weight_kg=75,
        sex="F",
        target_anxiolysis=80,
        target_sedation=30,
        target_amnesia=20,
        profile=PatientProfile.STANDARD,
        prefer_reversible=True,
    )

    report1 = advisor.analyze_patient(params1)
    advisor.print_report(report1)

    # Caso 2: Paciente geriátrico con insomnio
    print("\n" + "=" * 80)
    print("CASO 2: Paciente geriátrico - insomnio")
    print("=" * 80)

    params2 = PatientParameters(
        age=78,
        weight_kg=65,
        sex="M",
        target_anxiolysis=30,
        target_sedation=60,
        profile=PatientProfile.ELDERLY,
        avoid_amnesia=True,
        prefer_short_acting=True,
    )

    report2 = advisor.analyze_patient(params2)
    advisor.print_report(report2)

    # Caso 3: Sedación para procedimiento
    print("\n" + "=" * 80)
    print("CASO 3: Sedación procedural (endoscopia)")
    print("=" * 80)

    params3 = PatientParameters(
        age=55,
        weight_kg=80,
        sex="M",
        target_anxiolysis=60,
        target_sedation=70,
        target_amnesia=80,
        profile=PatientProfile.STANDARD,
        prefer_reversible=True,
    )

    report3 = advisor.analyze_patient(params3)
    advisor.print_report(report3)

    # Caso 4: Status epilepticus
    print("\n" + "=" * 80)
    print("CASO 4: Status epilepticus")
    print("=" * 80)

    params4 = PatientParameters(
        age=35,
        weight_kg=70,
        sex="F",
        target_anticonvulsant=90,
        target_sedation=50,
        profile=PatientProfile.STANDARD,
    )

    report4 = advisor.analyze_patient(params4)
    advisor.print_report(report4)

    # Exportar todos los casos
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    for i, report in enumerate([report1, report2, report3, report4], 1):
        filepath = output_dir / f"demo_case_{i}.json"
        advisor.export_report(report, str(filepath))

    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print(f"Reportes exportados a: {output_dir}")
    print("=" * 80)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Asesor Terapéutico Inverso - Sistema de evaluación"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecutar casos de demostración"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo de consulta"
    )

    args = parser.parse_args()

    if args.demo:
        demo_cases()
    elif args.interactive:
        interactive_consultation()
    else:
        # Por defecto, ejecutar demo
        demo_cases()
