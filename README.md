![Lion-logo.png](/Images/lion.png)

# LION: PET Tumor Segmentation

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/lionz?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=Downloads)](https://pepy.tech/projects/lionz)
[![DOI](https://zenodo.org/badge/685935027.svg)](https://zenodo.org/badge/latestdoi/685935027)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0)

LION segments tumors in PET scans. It supports FDG and PSMA tracers, runs on CPU/CUDA/Apple Silicon, and works as both a CLI tool and Python library.

## Quick Start

```bash
pip install lionz
lionz -d /path/to/data -m fdg
```

That's it. Models download automatically on first run.

## Requirements

- Python 3.10+
- 32GB RAM recommended
- GPU optional (NVIDIA CUDA or Apple Silicon MPS) - CPU works but slower

## Installation

**pip (recommended)**
```bash
pip install lionz
```

**From source with uv**
```bash
git clone https://github.com/QIMP-Team/LION.git
cd LION
uv sync
```

## Input Data Structure

```
data/
├── patient_001/
│   └── PT_scan.nii.gz      # PET file with PT_ prefix
├── patient_002/
│   └── PT_scan.nii.gz
└── patient_003/
    ├── PT_scan.nii.gz
    └── CT_scan.nii.gz      # CT optional, needs CT_ prefix
```

**Rules:**
- One folder per subject
- PET files must start with `PT_`
- CT files must start with `CT_` (optional)
- Supports `.nii` and `.nii.gz`
- DICOM folders also supported (modality auto-detected from tags)

### Messy Data?

Use the **lion-mcp** server with Claude Code or Codex to organize chaotic DICOM/NIfTI dumps. See [lion-mcp/README.md](lion-mcp/README.md).

## CLI Usage

```bash
# Basic
lionz -d /path/to/data -m fdg

# With SUV threshold
lionz -d /path/to/data -m fdg -t 2.5

# Generate MIP preview
lionz -d /path/to/data -m fdg -g

# Parallel processing (multiple subjects)
lionz -d /path/to/data -m fdg -p 4

# All options
lionz -d /path/to/data -m fdg -t 2.5 -g -p 4
```

**Options:**
| Flag | Description |
|------|-------------|
| `-d` | Input directory containing subject folders |
| `-m` | Model name: `fdg` or `psma` |
| `-t` | SUV threshold (requires SUV-calibrated input) |
| `-g` | Generate rotational MIP GIF |
| `-p` | Number of parallel jobs |
| `-h` | Show help |

## Library Usage

```python
import lionz

# From file path
result = lionz.lion('/path/to/PT_scan.nii.gz', 'fdg')

# From SimpleITK image
import SimpleITK as sitk
img = sitk.ReadImage('/path/to/PT_scan.nii.gz')
seg = lionz.lion(img, 'fdg')  # Returns SimpleITK.Image

# From numpy array
import numpy as np
array = np.load('pet_data.npy')
spacing = (2.0, 2.0, 2.0)
seg = lionz.lion((array, spacing), 'fdg')  # Returns np.ndarray

# With options
result = lionz.lion(
    '/path/to/scan.nii.gz',
    'fdg',
    output_dir='/path/to/output',
    accelerator='mps',  # 'cpu', 'cuda', or 'mps'
    threshold=2.5
)
```

**Important:** Wrap in `if __name__ == '__main__':` to avoid multiprocessing issues:

```python
import lionz

if __name__ == '__main__':
    lionz.lion('/path/to/scan.nii.gz', 'fdg')
```

## Models

| Tracer | Model | Training Data | Status |
|--------|-------|---------------|--------|
| FDG | `fdg` | 5,235 patients | Stable |
| PSMA | `psma` | 2,046 patients | Stable |

## Output Structure

```
patient_001/
├── PT_scan.nii.gz                    # Original input
└── lionz-2024-01-15-10-30-00/
    ├── segmentations/
    │   ├── PT_scan_tumor_seg.nii.gz  # Tumor mask
    │   └── patient_001_rotational_mip.gif  # If -g flag used
    └── stats/
        └── patient_001_metrics.csv   # Volume, SUV metrics
```

## Platform Support

| Platform | Accelerator | Status |
|----------|-------------|--------|
| Linux | CUDA | Fully supported |
| Linux | CPU | Supported (slower) |
| macOS (Apple Silicon) | MPS | Fully supported |
| macOS (Intel) | CPU | Supported (slower) |
| Windows | CUDA | Fully supported |
| Windows | CPU | Supported (slower) |

## For AI Agents

LION provides an MCP server for organizing messy medical imaging data. Install and configure:

```bash
cd lion-mcp && pip install -e .
```

Add to `.mcp.json`:
```json
{
  "lion-mcp": {
    "type": "stdio",
    "command": ".venv/bin/lion-mcp"
  }
}
```

**Available MCP tools:**
- `scan_directory` - Scan for DICOM/NIfTI files, extract metadata
- `read_dicom_header` - Read full DICOM tags
- `organize_for_lion` - Reorganize files into LION structure
- `validate_structure` - Check if directory is LION-ready
- `get_lion_requirements` - Get structure documentation

## Telemetry

LION collects anonymous usage statistics to help us understand how the tool is used and prioritize development. This is completely optional.

**What we collect:** version, model used, platform, accelerator type, number of subjects, success/failure

**What we DON'T collect:** file paths, patient data, IP addresses, any identifiable information

**Opt-out:**
```bash
export LIONZ_TELEMETRY=0
```

## Citation

```
DOI: 10.5281/zenodo.12626789
```

## License

Apache 2.0. For enterprise integrations, contact [Zenta](mailto:lalith@zenta.solutions).

## Contributors

- [Lalith Kumar Shiyam Sundar](https://github.com/LalithShiyam)
- [Manuel Pires](https://github.com/mprires)
- [Sebastian Gutschmayer](https://github.com/Keyn34)

Part of the [ENHANCE.PET](https://enhance.pet) initiative.
