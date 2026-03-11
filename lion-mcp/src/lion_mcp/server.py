#!/usr/bin/env python3
"""
LION MCP Server
---------------

MCP server providing tools for organizing messy DICOM/NIfTI dumps
into LION-compatible directory structure.

Tools:
- scan_directory: Scan for DICOM/NIfTI files and extract metadata
- read_dicom_header: Read detailed DICOM header for a specific file
- organize_for_lion: Move/copy files into LION-compatible structure
- validate_structure: Check if a directory is LION-ready
"""

import json
import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("lion-mcp")

# Constants
ALLOWED_MODALITIES = ["CT", "PT"]
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
DICOM_EXTENSIONS = {".dcm", ".dicom", ""}  # Empty string for extensionless DICOM


def is_dicom_file(filepath: str) -> bool:
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def is_nifti_file(filepath: str) -> bool:
    """Check if a file is a valid NIfTI file."""
    path = Path(filepath)
    if path.suffix == ".gz":
        return path.stem.endswith(".nii")
    return path.suffix == ".nii"


def remove_accents(text: str) -> str:
    """Remove accents and special characters from text."""
    try:
        text = str(text).replace(" ", "_")
        cleaned = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        cleaned = re.sub(r"[^\w\s-]", "", cleaned.strip().lower())
        cleaned = re.sub(r"[-\s]+", "-", cleaned)
        return cleaned
    except Exception:
        return text


def extract_dicom_metadata(filepath: str) -> dict[str, Any]:
    """Extract relevant metadata from a DICOM file."""
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

        metadata = {
            "filepath": filepath,
            "modality": getattr(ds, "Modality", None),
            "patient_id": getattr(ds, "PatientID", None),
            "patient_name": str(getattr(ds, "PatientName", "")) if hasattr(ds, "PatientName") else None,
            "study_date": getattr(ds, "StudyDate", None),
            "study_description": getattr(ds, "StudyDescription", None),
            "series_number": getattr(ds, "SeriesNumber", None),
            "series_description": getattr(ds, "SeriesDescription", None),
            "series_instance_uid": getattr(ds, "SeriesInstanceUID", None),
            "study_instance_uid": getattr(ds, "StudyInstanceUID", None),
            "sop_instance_uid": getattr(ds, "SOPInstanceUID", None),
            "manufacturer": getattr(ds, "Manufacturer", None),
            "institution_name": getattr(ds, "InstitutionName", None),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
        }

        # PET-specific fields
        if metadata["modality"] == "PT":
            metadata["units"] = getattr(ds, "Units", None)
            metadata["patient_weight_kg"] = getattr(ds, "PatientWeight", None)

            if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
                radio_seq = ds.RadiopharmaceuticalInformationSequence[0]
                metadata["radiopharmaceutical"] = getattr(radio_seq, "Radiopharmaceutical", None)
                metadata["radionuclide_total_dose_bq"] = getattr(radio_seq, "RadionuclideTotalDose", None)
                metadata["radionuclide_half_life_s"] = getattr(radio_seq, "RadionuclideHalfLife", None)
                metadata["radiopharmaceutical_start_time"] = getattr(radio_seq, "RadiopharmaceuticalStartTime", None)

        # CT-specific fields
        if metadata["modality"] == "CT":
            metadata["kvp"] = getattr(ds, "KVP", None)
            metadata["slice_thickness"] = getattr(ds, "SliceThickness", None)
            metadata["convolution_kernel"] = getattr(ds, "ConvolutionKernel", None)

        return metadata
    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


def extract_nifti_metadata(filepath: str) -> dict[str, Any]:
    """Extract metadata from a NIfTI file."""
    try:
        img = nib.load(filepath)
        header = img.header

        # Try to infer modality from filename
        filename = Path(filepath).name.upper()
        inferred_modality = None
        if "PT" in filename or "PET" in filename or "SUV" in filename:
            inferred_modality = "PT"
        elif "CT" in filename:
            inferred_modality = "CT"

        zooms = header.get_zooms()
        metadata = {
            "filepath": filepath,
            "format": "NIfTI",
            "inferred_modality": inferred_modality,
            "shape": list(img.shape),
            "voxel_spacing": [float(z) for z in zooms[:3]] if len(zooms) >= 3 else [float(z) for z in zooms],
            "data_dtype": str(header.get_data_dtype()),
            "affine": [[float(x) for x in row] for row in img.affine],
        }

        # Check for intensity range (useful for SUV)
        try:
            data = np.asarray(img.dataobj)
            metadata["intensity_min"] = float(np.min(data))
            metadata["intensity_max"] = float(np.max(data))
            metadata["intensity_mean"] = float(np.mean(data))
        except Exception:
            pass

        return metadata
    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


@mcp.tool()
def scan_directory(
    directory: str,
    recursive: bool = True,
    include_dicom: bool = True,
    include_nifti: bool = True,
) -> str:
    """
    Scan a directory for DICOM and NIfTI files, extracting metadata.

    This tool helps AI understand the structure of a messy medical imaging dump
    by identifying all imaging files and their key metadata (modality, patient info,
    series info, etc.).

    Args:
        directory: Path to the directory to scan
        recursive: Whether to scan subdirectories (default: True)
        include_dicom: Whether to include DICOM files (default: True)
        include_nifti: Whether to include NIfTI files (default: True)

    Returns:
        JSON string with scan results including file counts, series groupings,
        and metadata for each file.
    """
    directory = Path(directory)
    if not directory.exists():
        return json.dumps({"error": f"Directory does not exist: {directory}"})

    results = {
        "scan_time": datetime.now().isoformat(),
        "directory": str(directory),
        "recursive": recursive,
        "dicom_files": [],
        "nifti_files": [],
        "other_files": [],
        "dicom_series": {},  # Grouped by SeriesInstanceUID
        "summary": {
            "total_dicom": 0,
            "total_nifti": 0,
            "total_other": 0,
            "modalities_found": set(),
            "patients_found": set(),
        },
    }

    # Get all files
    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.glob("*"))

    files = [f for f in files if f.is_file()]

    for filepath in files:
        filepath_str = str(filepath)

        # Skip hidden files
        if filepath.name.startswith("."):
            continue

        # Check NIfTI first (by extension)
        if include_nifti and is_nifti_file(filepath_str):
            metadata = extract_nifti_metadata(filepath_str)
            results["nifti_files"].append(metadata)
            results["summary"]["total_nifti"] += 1
            if metadata.get("inferred_modality"):
                results["summary"]["modalities_found"].add(metadata["inferred_modality"])
            continue

        # Check DICOM
        if include_dicom and is_dicom_file(filepath_str):
            metadata = extract_dicom_metadata(filepath_str)
            results["dicom_files"].append(metadata)
            results["summary"]["total_dicom"] += 1

            # Group by series
            series_uid = metadata.get("series_instance_uid")
            if series_uid:
                if series_uid not in results["dicom_series"]:
                    results["dicom_series"][series_uid] = {
                        "modality": metadata.get("modality"),
                        "series_description": metadata.get("series_description"),
                        "series_number": metadata.get("series_number"),
                        "patient_id": metadata.get("patient_id"),
                        "study_instance_uid": metadata.get("study_instance_uid"),
                        "file_count": 0,
                        "sample_file": filepath_str,
                    }
                results["dicom_series"][series_uid]["file_count"] += 1

            if metadata.get("modality"):
                results["summary"]["modalities_found"].add(metadata["modality"])
            if metadata.get("patient_id"):
                results["summary"]["patients_found"].add(metadata["patient_id"])
            continue

        # Other files
        results["other_files"].append(filepath_str)
        results["summary"]["total_other"] += 1

    # Convert sets to lists for JSON serialization
    results["summary"]["modalities_found"] = list(results["summary"]["modalities_found"])
    results["summary"]["patients_found"] = list(results["summary"]["patients_found"])

    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def read_dicom_header(filepath: str, include_private_tags: bool = False) -> str:
    """
    Read the complete DICOM header for a specific file.

    Use this when you need detailed information about a specific DICOM file
    to make decisions about how to organize it.

    Args:
        filepath: Path to the DICOM file
        include_private_tags: Whether to include private/vendor-specific tags (default: False)

    Returns:
        JSON string with all DICOM header fields.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return json.dumps({"error": f"File does not exist: {filepath}"})

    try:
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)

        header = {}
        for elem in ds:
            # Skip pixel data
            if elem.tag == (0x7FE0, 0x0010):
                continue

            # Skip private tags unless requested
            if elem.tag.is_private and not include_private_tags:
                continue

            tag_name = elem.keyword if elem.keyword else f"({elem.tag.group:04X},{elem.tag.element:04X})"

            # Handle sequences
            if elem.VR == "SQ":
                seq_items = []
                for seq_item in elem.value:
                    seq_dict = {}
                    for seq_elem in seq_item:
                        seq_tag_name = seq_elem.keyword if seq_elem.keyword else f"({seq_elem.tag.group:04X},{seq_elem.tag.element:04X})"
                        try:
                            seq_dict[seq_tag_name] = str(seq_elem.value)
                        except Exception:
                            seq_dict[seq_tag_name] = "<unreadable>"
                    seq_items.append(seq_dict)
                header[tag_name] = seq_items
            else:
                try:
                    header[tag_name] = str(elem.value)
                except Exception:
                    header[tag_name] = "<unreadable>"

        return json.dumps(header, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def organize_for_lion(
    source_dir: str,
    target_dir: str,
    organization_plan: str,
    copy_files: bool = True,
    convert_dicom: bool = True,
) -> str:
    """
    Organize files into LION-compatible directory structure.

    This tool executes an organization plan created by the AI based on
    the scan results. The plan specifies how to group and rename files.

    LION-compatible structure:
    ```
    target_dir/
    ├── subject_001/
    │   ├── PT_scan.nii.gz    # PET image with PT_ prefix
    │   └── CT_scan.nii.gz    # CT image with CT_ prefix (optional)
    ├── subject_002/
    │   └── PT_scan.nii.gz
    └── ...
    ```

    Args:
        source_dir: Source directory containing the messy data
        target_dir: Target directory for organized output
        organization_plan: JSON string describing how to organize files.
            Format: {
                "subjects": [
                    {
                        "subject_id": "patient_001",
                        "files": [
                            {
                                "source": "/path/to/file.nii.gz",
                                "target_name": "PT_scan.nii.gz",
                                "modality": "PT"
                            }
                        ]
                    }
                ]
            }
        copy_files: If True, copy files; if False, move files (default: True)
        convert_dicom: If True, convert DICOM series to NIfTI (default: True)

    Returns:
        JSON string with organization results.
    """
    try:
        plan = json.loads(organization_plan)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in organization_plan: {e}"})

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    results = {
        "success": True,
        "organized_subjects": [],
        "errors": [],
        "warnings": [],
    }

    for subject in plan.get("subjects", []):
        subject_id = subject.get("subject_id", "unknown")
        subject_dir = target_path / subject_id
        subject_dir.mkdir(exist_ok=True)

        subject_result = {
            "subject_id": subject_id,
            "files_organized": [],
        }

        for file_spec in subject.get("files", []):
            source = Path(file_spec.get("source", ""))
            target_name = file_spec.get("target_name", "")
            modality = file_spec.get("modality", "")

            if not source.exists():
                results["errors"].append(f"Source file not found: {source}")
                continue

            # Ensure proper prefix
            if modality in ALLOWED_MODALITIES and not target_name.startswith(f"{modality}_"):
                target_name = f"{modality}_{target_name}"
                results["warnings"].append(f"Added modality prefix to {target_name}")

            # Ensure .nii.gz extension
            if not target_name.endswith((".nii", ".nii.gz")):
                target_name = f"{target_name}.nii.gz"

            target_file = subject_dir / target_name

            try:
                if source.is_dir() and convert_dicom:
                    # This is a DICOM directory - would need dicom2nifti
                    results["warnings"].append(
                        f"DICOM directory conversion not yet implemented: {source}"
                    )
                    continue

                if copy_files:
                    shutil.copy2(source, target_file)
                else:
                    shutil.move(source, target_file)

                subject_result["files_organized"].append({
                    "source": str(source),
                    "target": str(target_file),
                })
            except Exception as e:
                results["errors"].append(f"Failed to organize {source}: {e}")

        results["organized_subjects"].append(subject_result)

    if results["errors"]:
        results["success"] = False

    return json.dumps(results, indent=2)


@mcp.tool()
def validate_structure(directory: str) -> str:
    """
    Validate if a directory has LION-compatible structure.

    Checks:
    - Each subject folder contains NIfTI files with proper modality prefixes (PT_, CT_)
    - Files are readable and have valid headers
    - Required modalities are present (PT is required for LION)

    Args:
        directory: Path to the directory to validate

    Returns:
        JSON string with validation results including any issues found.
    """
    directory = Path(directory)
    if not directory.exists():
        return json.dumps({"error": f"Directory does not exist: {directory}"})

    results = {
        "valid": True,
        "directory": str(directory),
        "subjects": [],
        "summary": {
            "total_subjects": 0,
            "valid_subjects": 0,
            "subjects_missing_pt": 0,
        },
        "issues": [],
    }

    # Get all subdirectories (subjects)
    subjects = [d for d in directory.iterdir() if d.is_dir() and not d.name.startswith(".")]
    results["summary"]["total_subjects"] = len(subjects)

    for subject_dir in subjects:
        subject_result = {
            "subject_id": subject_dir.name,
            "valid": True,
            "has_pt": False,
            "has_ct": False,
            "files": [],
            "issues": [],
        }

        # Find NIfTI files
        nifti_files = list(subject_dir.glob("*.nii")) + list(subject_dir.glob("*.nii.gz"))

        if not nifti_files:
            subject_result["valid"] = False
            subject_result["issues"].append("No NIfTI files found")
            results["issues"].append(f"{subject_dir.name}: No NIfTI files found")

        for nifti_file in nifti_files:
            filename = nifti_file.name
            file_info = {"filename": filename, "valid": True, "issues": []}

            # Check modality prefix
            has_valid_prefix = False
            for modality in ALLOWED_MODALITIES:
                if filename.startswith(f"{modality}_"):
                    has_valid_prefix = True
                    if modality == "PT":
                        subject_result["has_pt"] = True
                    elif modality == "CT":
                        subject_result["has_ct"] = True
                    break

            if not has_valid_prefix:
                file_info["valid"] = False
                file_info["issues"].append(f"Missing modality prefix (expected PT_ or CT_)")
                subject_result["issues"].append(f"{filename}: Missing modality prefix")

            # Try to load the file
            try:
                img = nib.load(str(nifti_file))
                file_info["shape"] = list(img.shape)
                file_info["spacing"] = [float(s) for s in img.header.get_zooms()[:3]]
            except Exception as e:
                file_info["valid"] = False
                file_info["issues"].append(f"Failed to load: {e}")
                subject_result["issues"].append(f"{filename}: Failed to load")

            subject_result["files"].append(file_info)

        # PT is required for LION
        if not subject_result["has_pt"]:
            subject_result["valid"] = False
            subject_result["issues"].append("Missing required PT (PET) image")
            results["issues"].append(f"{subject_dir.name}: Missing required PT image")
            results["summary"]["subjects_missing_pt"] += 1

        if subject_result["valid"]:
            results["summary"]["valid_subjects"] += 1
        else:
            results["valid"] = False

        results["subjects"].append(subject_result)

    return json.dumps(results, indent=2)


@mcp.tool()
def get_lion_requirements() -> str:
    """
    Get the requirements for LION-compatible directory structure.

    Returns documentation on how data should be organized for LION processing.

    Returns:
        JSON string with LION requirements and examples.
    """
    requirements = {
        "description": "LION (Lesion segmentatION) requires a specific directory structure for processing PET/CT scans.",
        "structure": {
            "pattern": "parent_directory/subject_folder/modality_image.nii.gz",
            "example": [
                "data/",
                "├── patient_001/",
                "│   ├── PT_fdg_scan.nii.gz    # Required: PET image",
                "│   └── CT_scan.nii.gz         # Optional: CT image",
                "├── patient_002/",
                "│   └── PT_psma_scan.nii.gz",
                "└── patient_003/",
                "    ├── PT_scan.nii.gz",
                "    └── CT_scan.nii.gz",
            ],
        },
        "file_requirements": {
            "format": "NIfTI (.nii or .nii.gz)",
            "naming": "Must start with modality prefix: PT_ for PET, CT_ for CT",
            "required_modality": "PT (PET) - at least one PT file per subject",
            "optional_modality": "CT - for anatomical reference",
        },
        "pet_requirements": {
            "units": "Should be in SUV (Standardized Uptake Value)",
            "note": "DICOM PET images in Bq/mL or counts will be automatically converted to SUV during DICOM-to-NIfTI conversion if proper DICOM headers are present",
        },
        "available_models": {
            "fdg": "FDG-PET tumor segmentation (trained on 5341 cases)",
            "psma": "PSMA-PET tumor segmentation (trained on 2299 cases)",
        },
        "cli_usage": "lionz -d /path/to/data -m fdg",
        "threshold_note": "The -t threshold option filters by SUV value",
    }

    return json.dumps(requirements, indent=2)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
