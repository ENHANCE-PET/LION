# LION MCP Server

MCP server for organizing messy DICOM/NIfTI dumps into LION-compatible directory structure. Designed for use with Claude Code, Codex, or any MCP-compatible client.

## Installation

```bash
# From the lion repo root, with venv activated
cd lion-mcp
uv pip install -e .
```

## Usage

### With Claude Code

The lion repo includes a `.mcp.json` that automatically configures the server. When you open Claude Code in the lion directory, approve the MCP server when prompted.

### Manual Configuration

Add to your project's `.mcp.json`:

```json
{
  "lion-mcp": {
    "type": "stdio",
    "command": "/path/to/lion/.venv/bin/lion-mcp"
  }
}
```

### Running Standalone

```bash
# Test the server directly
lion-mcp
```

## Available Tools

### `scan_directory`
Scan a directory for DICOM/NIfTI files and extract metadata.

```
Arguments:
  - directory: Path to scan
  - recursive: Scan subdirectories (default: true)
  - include_dicom: Include DICOM files (default: true)
  - include_nifti: Include NIfTI files (default: true)
```

### `read_dicom_header`
Read the complete DICOM header for a specific file.

```
Arguments:
  - filepath: Path to the DICOM file
  - include_private_tags: Include vendor-specific tags (default: false)
```

### `organize_for_lion`
Move/copy files into LION-compatible directory structure.

```
Arguments:
  - source_dir: Source directory with messy data
  - target_dir: Target directory for organized output
  - organization_plan: JSON describing how to organize (see below)
  - copy_files: Copy instead of move (default: true)
```

### `validate_structure`
Check if a directory has LION-compatible structure.

```
Arguments:
  - directory: Path to validate
```

### `get_lion_requirements`
Get documentation on LION structure requirements.

## LION-Compatible Structure

```
data/
├── patient_001/
│   ├── PT_fdg_scan.nii.gz    # Required: PET with PT_ prefix
│   └── CT_scan.nii.gz         # Optional: CT with CT_ prefix
├── patient_002/
│   └── PT_psma_scan.nii.gz
└── ...
```

**Requirements:**
- Files must be NIfTI format (`.nii` or `.nii.gz`)
- PET files must have `PT_` prefix
- CT files must have `CT_` prefix
- Each subject folder must have at least one PT file

## Example Workflow

1. Scan the messy directory:
   ```
   Use scan_directory on /path/to/messy/dump
   ```

2. AI analyzes the results and creates an organization plan

3. Execute the organization:
   ```
   Use organize_for_lion with the plan
   ```

4. Validate the result:
   ```
   Use validate_structure on /path/to/output
   ```

## Organization Plan Format

```json
{
  "subjects": [
    {
      "subject_id": "patient_001",
      "files": [
        {
          "source": "/path/to/original/pet_image.nii.gz",
          "target_name": "PT_fdg_scan.nii.gz",
          "modality": "PT"
        },
        {
          "source": "/path/to/original/ct_image.nii.gz",
          "target_name": "CT_scan.nii.gz",
          "modality": "CT"
        }
      ]
    }
  ]
}
```
