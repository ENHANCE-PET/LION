#!/usr/bin/env python3
import os
import argparse
from lionz import lion

def process_all(input_root, output_dir, model, accelerator):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all files in the input root
    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                input_path = os.path.join(dirpath, fname)
                output_path = os.path.join(output_dir, fname)
                print(f"\n🔹 Processing: {input_path}")
                try:
                    lion(input_path, model, output_path, accelerator)
                    print(f"✅ Saved: {output_path}")
                except Exception as e:
                    print(f"❌ Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch-process NIfTI files with lionz.lion"
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Path to the root directory containing .nii or .nii.gz files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to directory where outputs will be saved",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use (e.g., 'fdg', 'psma', etc.)",
    )
    parser.add_argument(
        "--accelerator",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute accelerator to use (default: cuda)",
    )

    args = parser.parse_args()
    process_all(args.input_root, args.output_dir, args.model, args.accelerator)

if __name__ == "__main__":
    main()