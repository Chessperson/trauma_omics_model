"""
Environment diagnostic — run this first before anything else.
Just run: python check_environment.py
"""

import sys
print("=" * 50)
print("PYTHON")
print("=" * 50)
print(f"Version: {sys.version}")

print()
print("=" * 50)
print("CORE LIBRARIES")
print("=" * 50)

libs = {
    "torch":        "PyTorch",
    "numpy":        "NumPy",
    "pandas":       "Pandas",
    "sklearn":      "Scikit-learn",
    "scipy":        "SciPy",
    "openpyxl":     "OpenPyXL (Excel reading)",
    "matplotlib":   "Matplotlib",
}

missing = []
for lib, name in libs.items():
    try:
        mod = __import__(lib)
        version = getattr(mod, "__version__", "installed")
        print(f"  [OK]  {name}: {version}")
    except ImportError:
        print(f"  [!!]  {name}: NOT INSTALLED")
        missing.append(lib)

print()
print("=" * 50)
print("PYTORCH + CUDA")
print("=" * 50)

try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  GPU device      : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  Running on      : CPU only")
        print("  NOTE: Training will be slow on CPU — GPU strongly recommended for transformers")
except ImportError:
    print("  [!!] PyTorch not installed")

print()
print("=" * 50)
print("FILE CHECK")
print("=" * 50)

import os
files = [
    "PAMPer_External_data.xlsx",
    "synthetic_PAMPer_External_full.xlsx",
    "PAMPer_Metabolon_metabolomics.xlsx",
    "synthetic_metabolomics.xlsx",
]

for f in files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1e6
        print(f"  [OK]  {f} ({size_mb:.1f} MB)")
    else:
        print(f"  [!!]  {f}: NOT FOUND in current directory")

print()
print("=" * 50)
if missing:
    print("INSTALL MISSING LIBRARIES WITH:")
    print("=" * 50)
    pip_names = {
        "torch": "torch torchvision",
        "sklearn": "scikit-learn",
    }
    for lib in missing:
        pkg = pip_names.get(lib, lib)
        print(f"  pip install {pkg}")
else:
    print("ALL LIBRARIES INSTALLED — you are good to go!")
    print("=" * 50)