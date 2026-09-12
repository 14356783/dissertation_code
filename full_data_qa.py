"""
APPENDIX: PHITS T-SED Data Quality Assurance (QA) & Microdosimetric Metric Extraction
========================================================================================
This script batch-parses PHITS T-SED output files for various ion species and energies.
It validates energy deposition spectra against statistical convergence criteria, checks
for spectral gaps, and calculates key microdosimetric mean parameters:
  - Frequency-mean lineal energy (y_F)
  - Dose-mean lineal energy (y_D)
"""

import glob
import os
import re
import numpy as np

# Regular expression to parse particle species and kinetic energy from filenames
# Matches formats like: edep_proton_10mev.out, edep_4he_10mevn.out, edep_56fe_65mevn.out
FILENAME_RE = re.compile(
    r"edep_(?P<particle>[A-Za-z0-9]+)_(?P<energy>[\d.]+)mevn?\.out$"
)


def parse_filename(path):
    """
    Extracts particle species name and incident kinetic energy (MeV or MeV/n)
    from standard PHITS filename patterns.
    """
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return m.group("particle"), float(m.group("energy"))


# Search current directory for all matching PHITS output tally files
candidates = glob.glob("edep_*.out")
labelled = []
for f in candidates:
    particle, energy = parse_filename(f)
    if particle is None:
        continue
    labelled.append((particle, energy, f))

# Sort runs numerically by energy, then alphabetically by particle species
labelled.sort(key=lambda t: (t[1], t[0]))

# Formatted Console Summary Table Header
header = (
    f"{'Energy Run':<16} | "
    f"{'Total Err':<9} | "
    f"{'Err N-1':<8} | "
    f"{'Err N':<8} | "
    f"{'Gaps?':<6} | "
    f"{'y_F':<8} | "
    f"{'y_D':<8} | "
    f"{'QA Status'}"
)

print("=" * len(header))
print(header)
print("=" * len(header))

# Geometric Mean Chord Length for a 2 µm cubic Sensitive Volume:
# l_bar = (2/3) * d = (2/3) * 2.0 µm = 1.3333 µm
L_BAR = 1.3333

for particle, energy, file_path in labelled:
    label = f"{particle}_{energy:g}mevn"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            line_str = line.strip()
            # Ignore PHITS header metadata lines and standard comment markers
            if (
                line_str
                and not line_str.startswith("#")
                and not line_str.startswith("c")
            ):
                parts = line_str.split()
                if len(parts) >= 3:
                    try:
                        # Extract: [Energy Bin Boundary (MeV), Bin Yield, Relative Statistical Error]
                        data.append(
                            [
                                float(parts[0]),
                                float(parts[-2]),
                                float(parts[-1]),
                            ]
                        )
                    except ValueError:
                        continue

    if not data:
        continue

    arr = np.array(data)
    energies = arr[:, 0]     # Energy deposited (MeV)
    values = arr[:, 1]       # Bin yield / tally value
    rel_errors = arr[:, 2]   # Relative statistical uncertainty (fraction)

    # Convert Deposited Energy (MeV) to Lineal Energy y (keV/µm)
    # y = E_dep / l_bar
    y_values = (energies * 1000.0) / L_BAR

    # Mask zero-yield bins to isolate statistical errors on actual data points
    valid_mask = values > 0
    valid_err = rel_errors[valid_mask] * 100 if np.any(valid_mask) else np.array([])

    # 1. Metric: Average relative error across all non-zero spectrum bins (%)
    total_avg_err = np.mean(valid_err) if len(valid_err) > 0 else 0.0

    # 2. Metric: Extract relative errors of the highest energy tail bins (N-1 and N)
    if len(valid_err) >= 2:
        err_n1 = valid_err[-2]
        err_n = valid_err[-1]
    elif len(valid_err) == 1:
        err_n1 = 0.0
        err_n = valid_err[-1]
    else:
        err_n1, err_n = 0.0, 0.0

    # 3. Metric: Spectral Gap Check (Flags unphysical zero-yield bins between active spectrum limits)
    non_zero_indices = np.where(values > 0)[0]
    if len(non_zero_indices) > 0:
        first_idx, last_idx = non_zero_indices[0], non_zero_indices[-1]
        has_gaps = np.any(values[first_idx:last_idx] == 0)
    else:
        has_gaps = True

    # 4. Microdosimetric Parameter Extraction (y_F and y_D)
    dy = np.gradient(y_values)
    norm = np.sum(values * dy)

    if norm > 0:
        # Frequency-mean lineal energy (keV/µm)
        y_F = np.sum(y_values * values * dy) / norm
        # Dose-mean lineal energy (keV/µm)
        y_D = np.sum((y_values**2) * values * dy) / np.sum(y_values * values * dy)
    else:
        y_F, y_D = 0.0, 0.0

    # 5. Data Quality Assurance Evaluation Logic
    if has_gaps:
        qa_status = "FLAGGED"
    elif err_n1 <= 20.0:
        if err_n <= 50.0:
            qa_status = "PASSED"
        else:
            qa_status = "PASSED (Tail Spike)"
    else:
        qa_status = "ACCEPTABLE"

    gap_str = "YES" if has_gaps else "No"

    # Print clean formatted summary row
    print(
        f"{label:<16} | "
        f"{total_avg_err:7.2f}% | "
        f"{err_n1:6.2f}% | "
        f"{err_n:6.2f}% | "
        f"{gap_str:<6} | "
        f"{y_F:8.2f} | "
        f"{y_D:8.2f} | "
        f"{qa_status}"
    )

print("=" * len(header))