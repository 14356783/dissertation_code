"""
APPENDIX: Automated PHITS Batch Execution and Parameter Injection Script
========================================================================
This script dynamically generates energy- and particle-specific PHITS input cards 
from master template files, manages random number seeding, executes PHITS in isolated 
run directories, and routes the resulting T-SED output files to a central directory.
"""

import os
import random
import shutil
import subprocess
import sys

# Particle species unit mapping (protons use MeV; heavier ions use MeV/n)
UNIT_SUFFIX = {
    "proton": "mev",
    "4he": "mevn",
    "56fe": "mevn",
    "12c": "mevn",
    "16o": "mevn",
}

# CLI Argument Validation
if len(sys.argv) != 5:
    print("Usage: python3 run_energy.py <particle> <energy> <maxcas> <maxbch>")
    print(f"  particle: one of {list(UNIT_SUFFIX.keys())}")
    sys.exit(1)

particle = sys.argv[1]
E = sys.argv[2]
current_maxcas = int(sys.argv[3])
current_maxbch = int(sys.argv[4])

if particle not in UNIT_SUFFIX:
    print(f"Unknown particle '{particle}'. Known: {list(UNIT_SUFFIX.keys())}")
    sys.exit(1)

unit = UNIT_SUFFIX[particle]

# Generate pseudo-random seed for PHITS Monte Carlo transport (32-bit positive integer)
current_rseed = random.SystemRandom().randint(1, 2_147_483_647)

# Configure OpenMP thread binding environment
phits_env = os.environ.copy()
phits_env["OMP_PROC_BIND"] = "close"
phits_env["OMP_PLACES"] = "cores"
phits_bin = "phits"

# Establish target output and temporary working directories
outputs_dir = os.path.abspath("outputs")
os.makedirs(outputs_dir, exist_ok=True)

workdir = os.path.abspath(f"run_{E}{unit}")
os.makedirs(workdir, exist_ok=True)
os.makedirs(os.path.join(workdir, "outputs"), exist_ok=True)

print(
    f"--> Running {particle}  E={E}{unit}  maxcas={current_maxcas} "
    f" maxbch={current_maxbch}  rseed={current_rseed}"
)

# Template path definitions
template_main = f"template_{particle}.inp"
template_source = f"template_{particle}_source.inp"

# 1. Inject particle kinetic energy into source template card
with open(template_source, "r") as f:
    source_content = f.read().replace("__ENERGY__", str(E))
source_path = os.path.join(workdir, f"source_{particle}_{E}{unit}.inp")
with open(source_path, "w") as f:
    f.write(source_content)

# 2. Inject parameters (Energy, Max Cases, Max Batches, Seed) into main template card
with open(template_main, "r") as f:
    main_content = (
        f.read()
        .replace("__ENERGY__", str(E))
        .replace("__MAXCAS__", str(current_maxcas))
        .replace("__MAXBCH__", str(current_maxbch))
        .replace("__RSEED__", str(current_rseed))
    )
main_path = os.path.join(workdir, f"main_{E}{unit}.inp")
with open(main_path, "w") as f:
    f.write(main_content)

# 3. Execute PHITS binary within dedicated working directory
result = subprocess.run([phits_bin, f"main_{E}{unit}.inp"], cwd=workdir, env=phits_env)

# 4. Verification and T-SED tally extraction
tally_output = os.path.join(
    workdir, "outputs", f"edep_{particle}_{E}{unit}.out"
)
if os.path.exists(tally_output):
    dest = os.path.join(outputs_dir, f"edep_{particle}_{E}{unit}.out")
    shutil.copy(tally_output, dest)
    print(f"OK      {particle} E={E}{unit}  seed={current_rseed}")
    os.remove(main_path)
    os.remove(source_path)
else:
    print(
        f"MISSING {particle} E={E}{unit}  seed={current_rseed}  (expected"
        f" {tally_output}, returncode={result.returncode})"
    )
    sys.exit(1)