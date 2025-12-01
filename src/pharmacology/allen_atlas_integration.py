#!/usr/bin/env python3
"""
Allen Brain Atlas Integration Module
=====================================

Integrates receptor density data from Allen Human Brain Atlas
into the brain receptor map for more accurate regional modeling.

Author: Francisco Molina Burgos (Yatrogenesis)
ORCID: 0009-0008-6093-8267

References:
- Hawrylycz MJ et al (2012) An anatomically comprehensive atlas of the adult
  human brain transcriptome. Nature 489:391-399
- https://human.brain-map.org/
"""

import json
from pathlib import Path
from typing import Dict, Optional


def load_allen_atlas_densities(atlas_file: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Load receptor densities from Allen Brain Atlas data.

    Args:
        atlas_file: Path to the receptor densities JSON file.
                   If None, uses default location.

    Returns:
        Dict mapping receptor gene symbol -> {region: density}
    """
    if atlas_file is None:
        atlas_file = Path(__file__).parent.parent.parent / "data" / "allen_atlas" / "receptor_densities_literature.json"
    else:
        atlas_file = Path(atlas_file)

    if not atlas_file.exists():
        print(f"Warning: Allen Atlas file not found at {atlas_file}")
        return {}

    with open(atlas_file, 'r') as f:
        data = json.load(f)

    return data.get('receptors', {})


def get_region_mapping() -> Dict[str, list]:
    """
    Get mapping from simplified brain regions to Allen Atlas regions.

    Returns:
        Dict mapping our region names to lists of Allen region names
    """
    return {
        'cortex': ['prefrontal_cortex', 'motor_cortex', 'somatosensory_cortex'],
        'hippocampus': ['hippocampus'],
        'striatum': ['caudate', 'putamen', 'nucleus_accumbens'],
        'thalamus': ['thalamus'],
        'brainstem': ['brainstem', 'locus_coeruleus', 'raphe_nuclei'],
        'amygdala': ['amygdala'],
        'hypothalamus': ['hypothalamus'],
        'cerebellum': ['cerebellum'],
        'vta': ['ventral_tegmental_area'],
        'snc': ['substantia_nigra'],
    }


def get_density_for_region(
    gene_symbol: str,
    region: str,
    atlas_data: Dict[str, Dict[str, float]]
) -> Optional[float]:
    """
    Get receptor density for a specific brain region.

    Args:
        gene_symbol: Gene symbol (e.g., 'GABRA1')
        region: Our simplified region name (e.g., 'cortex')
        atlas_data: Loaded Allen Atlas data

    Returns:
        Normalized density (0-1) or None if not available
    """
    if gene_symbol not in atlas_data:
        return None

    receptor_data = atlas_data[gene_symbol]
    region_mapping = get_region_mapping()

    if region not in region_mapping:
        return None

    allen_regions = region_mapping[region]
    values = [receptor_data.get(r, 0) for r in allen_regions if r in receptor_data]

    if not values:
        return None

    return sum(values) / len(values)


def get_all_densities_for_receptor(
    gene_symbol: str,
    atlas_data: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Get all regional densities for a receptor.

    Args:
        gene_symbol: Gene symbol (e.g., 'GABRA1')
        atlas_data: Loaded Allen Atlas data

    Returns:
        Dict mapping region -> density
    """
    result = {}
    for region in get_region_mapping().keys():
        density = get_density_for_region(gene_symbol, region, atlas_data)
        if density is not None:
            result[region] = density
    return result


def compare_receptor_densities(
    gene1: str,
    gene2: str,
    atlas_data: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare regional densities between two receptors.

    Args:
        gene1: First gene symbol
        gene2: Second gene symbol
        atlas_data: Loaded Allen Atlas data

    Returns:
        Dict with comparison data
    """
    d1 = get_all_densities_for_receptor(gene1, atlas_data)
    d2 = get_all_densities_for_receptor(gene2, atlas_data)

    regions = set(d1.keys()) | set(d2.keys())

    comparison = {}
    for region in regions:
        v1 = d1.get(region, 0)
        v2 = d2.get(region, 0)
        comparison[region] = {
            gene1: v1,
            gene2: v2,
            'ratio': v1 / v2 if v2 > 0 else float('inf'),
            'difference': v1 - v2,
        }

    return comparison


def get_receptor_regional_profile(
    gene_symbol: str,
    atlas_data: Dict[str, Dict[str, float]]
) -> str:
    """
    Get a text description of regional receptor distribution.

    Args:
        gene_symbol: Gene symbol
        atlas_data: Loaded Allen Atlas data

    Returns:
        Human-readable profile description
    """
    densities = get_all_densities_for_receptor(gene_symbol, atlas_data)

    if not densities:
        return f"No data available for {gene_symbol}"

    # Sort by density
    sorted_regions = sorted(densities.items(), key=lambda x: -x[1])

    # Classify
    high = [r for r, d in sorted_regions if d >= 0.7]
    medium = [r for r, d in sorted_regions if 0.4 <= d < 0.7]
    low = [r for r, d in sorted_regions if d < 0.4]

    profile = f"{gene_symbol} Distribution:\n"
    if high:
        profile += f"  HIGH (>70%): {', '.join(high)}\n"
    if medium:
        profile += f"  MEDIUM (40-70%): {', '.join(medium)}\n"
    if low:
        profile += f"  LOW (<40%): {', '.join(low)}\n"

    return profile


def demo_allen_integration():
    """Demonstrate Allen Atlas integration."""
    print("=" * 70)
    print("ALLEN BRAIN ATLAS INTEGRATION")
    print("=" * 70)

    atlas_data = load_allen_atlas_densities()

    if not atlas_data:
        print("No Allen Atlas data found. Run download_allen_atlas.py first.")
        return

    print(f"\nLoaded data for {len(atlas_data)} receptors")

    # Show profiles for key receptors
    print("\n--- GABA-A SUBUNITS ---")
    for gene in ['GABRA1', 'GABRA2', 'GABRA5']:
        if gene in atlas_data:
            print(get_receptor_regional_profile(gene, atlas_data))

    print("\n--- DOPAMINE RECEPTORS ---")
    for gene in ['DRD1', 'DRD2']:
        if gene in atlas_data:
            print(get_receptor_regional_profile(gene, atlas_data))

    print("\n--- NMDA RECEPTORS ---")
    for gene in ['GRIN2B']:
        if gene in atlas_data:
            print(get_receptor_regional_profile(gene, atlas_data))

    print("\n--- SEROTONIN RECEPTORS ---")
    for gene in ['HTR1A', 'HTR2A']:
        if gene in atlas_data:
            print(get_receptor_regional_profile(gene, atlas_data))

    # Compare alpha-1 vs alpha-5 (sedation vs memory)
    print("\n--- GABRA1 vs GABRA5 COMPARISON ---")
    print("(Alpha-1 = sedation, Alpha-5 = memory/cognition)")
    comparison = compare_receptor_densities('GABRA1', 'GABRA5', atlas_data)
    for region, data in sorted(comparison.items()):
        print(f"  {region}: α1={data['GABRA1']:.2f} α5={data['GABRA5']:.2f} ratio={data['ratio']:.2f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo_allen_integration()
