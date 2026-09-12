"""
APPENDIX: PHITS T-SED Output Parsing and Orbital Rate Integration
================================================================
This script extracts energy deposition spectra (T-SED) from PHITS Monte Carlo
simulations, calculates effective Single Event Upset (SEU) cross-sections per bit,
propagates statistical uncertainties, and folds cross-sections with SPENVIS differential
flux data.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


def parse_phits_tsed(filepath):
    """
    Parses a PHITS T-SED output file to extract energy deposition spectrum.
    
    Returns DataFrame containing bin boundaries (sed_lwr, sed_upr in MeV),
    bin yield (value in deposit/source), and relative statistical error (r.err).
    """
    sed_lwr, sed_upr, values, errors = [], [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            # Skip PHITS header lines and comment characters
            if not line_stripped or line_stripped.startswith(('#', "'", 'c', 'msuc')):
                continue
            parts = line_stripped.split()
            if len(parts) >= 4:
                try:
                    sed_lwr.append(float(parts[0]))
                    sed_upr.append(float(parts[1]))
                    values.append(float(parts[2]))
                    errors.append(float(parts[3]))
                except ValueError:
                    continue

    return pd.DataFrame({
        'sed_lwr': sed_lwr,
        'sed_upr': sed_upr,
        'value': values,
        'r.err': errors,
    })


def weibull_cross_section(
    let,
    let_th=5.0,        # L0 threshold (MeV*cm^2/mg)
    sigma_sat=1.8e-6,  # Saturation cross-section (cm^2/bit)
    width=14.0,        # Weibull width parameter w (MeV*cm^2/mg)
    exponent=1.9,      # Weibull shape parameter s
):
    """Calculates heavy-ion Weibull SEU response for a given LET value."""
    let = np.atleast_1d(let)
    sigma = np.zeros_like(let, dtype=float)

    mask = let > let_th
    sigma[mask] = sigma_sat * (1.0 - np.exp(-((let[mask] - let_th) / width) ** exponent))
    return sigma


def calculate_proton_cross_section(df, max_rel_err=0.50):
    """
    Calculates effective proton cross section per bit and propagates statistical errors.
    
    Parameters:
        df (DataFrame): PHITS output data from parse_phits_tsed.
        max_rel_err (float): Statistical uncertainty cutoff filter (default 50%).
    """
    if df is None or df.empty:
        return 0.0, 0.0, 0.0

    # Filter out statistically noisy bins above threshold
    df_clean = df[df['r.err'] <= max_rel_err].copy()
    if df_clean.empty:
        return 0.0, 0.0, 0.0

    # Geometric mean for energy bins (MeV)
    e_mid = np.sqrt(df_clean['sed_lwr'] * df_clean['sed_upr'])
    de = df_clean['sed_upr'] - df_clean['sed_lwr']

    # Target Geometry & Physical Constants
    AREAL_DENSITY = 0.466  # mg/cm^2 (2 um Silicon Sensitive Volume, rho = 2.33 g/cm^3)
    A_PHITS_CELL = 4.0e-8  # PHITS target surface area (cm^2)

    # Convert Deposited Energy (MeV) to Linear Energy Transfer (LET) equivalent
    let_values = e_mid / AREAL_DENSITY

    # Heavy-ion response weighting per energy deposition bin
    sigma_ion = weibull_cross_section(let_values)
    bin_weights = (sigma_ion * de) / A_PHITS_CELL

    # Yields and bin uncertainties
    y_i = df_clean['value']
    delta_y_i = y_i * df_clean['r.err']

    # Integrated Effective Cross Section (cm^2/bit)
    proton_cs = np.sum(y_i * bin_weights)

    # Quadrature Error Propagation: sqrt(sum((w_i * Delta Y_i)^2))
    abs_err_cs = np.sqrt(np.sum((bin_weights * delta_y_i) ** 2))
    rel_err_cs = (abs_err_cs / proton_cs) if proton_cs > 0 else 0.0

    return max(0.0, proton_cs), abs_err_cs, rel_err_cs


def parse_spenvis_flux(filepath):
    """
    Parses SPENVIS differential flux output files and converts raw units
    (m^-2 s^-1 sr^-1 MeV^-1) to 4pi omnidirectional flux (cm^-2 s^-1 MeV^-1).
    """
    energies, dflux_raw = [], []

    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith(("'", '#', 'c', 'C', 'Average', 'End')):
                continue

            clean_line = line_str.replace(',', ' ')
            parts = clean_line.split()
            if len(parts) >= 3:
                try:
                    energies.append(float(parts[0]))
                    dflux_raw.append(float(parts[2]))
                except ValueError:
                    continue

    df_spenvis = pd.DataFrame({'energy_MeV': energies, 'dflux_raw': dflux_raw})
    if df_spenvis.empty:
        return df_spenvis

    df_spenvis = df_spenvis.dropna().sort_values('energy_MeV').reset_index(drop=True)

    # Unit Conversion Factor: m^-2 to cm^-2 (1e-4) * 4pi omnidirectional geometry
    SPENVIS_UNIT_CONVERSION = 1e-4 * (4.0 * np.pi)
    df_spenvis['dflux_cm2'] = df_spenvis['dflux_raw'] * SPENVIS_UNIT_CONVERSION

    return df_spenvis


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    ERROR_CUTOFF = 0.50  # 50% max relative statistical error threshold
    energies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 20, 22, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 170, 190, 200]
    
    # Step 1: Process PHITS Simulation Output Bins
    proton_cross_sections = {}
    for energy in energies:
        filename = f"edep_proton_{energy}mev.out"
        if os.path.exists(filename):
            df_phits = parse_phits_tsed(filename)
            cs, abs_err, rel_err = calculate_proton_cross_section(df_phits, max_rel_err=ERROR_CUTOFF)
            proton_cross_sections[energy] = {'cs': cs, 'abs_err': abs_err, 'rel_err': rel_err}

    # Step 2: Perform Orbital Flux Folding
    spenvis_filename = "spenvis_proton_flux.txt"
    if os.path.exists(spenvis_filename) and proton_cross_sections:
        sim_energies = np.array(list(proton_cross_sections.keys()))
        sim_sigma = np.array([v['cs'] for v in proton_cross_sections.values()])
        sim_abs_err = np.array([v['abs_err'] for v in proton_cross_sections.values()])

        spenvis_df = parse_spenvis_flux(spenvis_filename)
        flux_energies = spenvis_df['energy_MeV'].values
        differential_flux = spenvis_df['dflux_cm2'].values

        # Linear Interpolation of Cross-Section and Error Functions
        sigma_interp = interp1d(sim_energies, sim_sigma, kind='linear', bounds_error=False, fill_value=(0.0, sim_sigma[-1]))
        abs_err_interp = interp1d(sim_energies, sim_abs_err, kind='linear', bounds_error=False, fill_value=(0.0, sim_abs_err[-1]))

        interp_sigma = np.maximum(0.0, sigma_interp(flux_energies))
        interp_abs_err = np.maximum(0.0, abs_err_interp(flux_energies))

        # Orbital Upset Rate Integration (Trapezoidal)
        total_rate_sec = trapezoid(interp_sigma * differential_flux, flux_energies)

        # Discrete Trapezoidal Variance Propagation across Flux Spectrum
        dE = np.diff(flux_energies)
        f_err = interp_abs_err * differential_flux
        var_components = 0.25 * (dE**2) * (f_err[:-1]**2 + f_err[1:]**2)
        abs_err_rate_sec = np.sqrt(np.sum(var_components))

        rel_err_rate = (abs_err_rate_sec / total_rate_sec) if total_rate_sec > 0 else 0.0
        rate_per_day = total_rate_sec * 86400.0
        abs_err_rate_day = abs_err_rate_sec * 86400.0

        if rate_per_day > 0:
            mtbf_years = 1.0 / (rate_per_day * 365.25)
        else:
            mtbf_years = float("inf")

        print(f"Final Orbital Rate: {total_rate_sec:.5e} +/- {abs_err_rate_sec:.5e} err/bit/s ({rel_err_rate:.1%})")
        print(f"                  : {rate_per_day:.5e} +/- {abs_err_rate_day:.5e} err/bit/day")
        print(f"MTBF              : {mtbf_years:,.1f} bit-years")

        report_filename = "proton_orbital_rate_summary.txt"
        with open(report_filename, "w") as f:
            f.write("=== SEE Orbital Rate Analysis Summary ===\n")
            f.write("Primary Ion: Proton\n")
            f.write(f"Statistical Noise Filter: Bins with r.err > {ERROR_CUTOFF:.0%} excluded\n")
            f.write(f"Environment File: {spenvis_filename}\n")
            f.write("Target Geometry: 2 um Silicon Cuboid (rho = 2.33 g/cm^3)\n\n")
            f.write(f"Total Orbital Upset Rate: {total_rate_sec:.5e} ± {abs_err_rate_sec:.5e} errors/bit/s ({rel_err_rate:.1%})\n")
            f.write(f"                        : {rate_per_day:.5e} ± {abs_err_rate_day:.5e} errors/bit/day\n")
            f.write(f"Mean Time Between Failures: {mtbf_years:,.1f} bit-years\n")