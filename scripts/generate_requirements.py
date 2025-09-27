#!/usr/bin/env python3
"""
Script to generate various requirements files from pyproject.toml
Generates requirements.txt files for different CUDA versions and configurations.
"""

import subprocess
import sys
from pathlib import Path


def generate_requirements_files():
    """Generate all requirements files."""
    # Get the project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent

    # Change to project root directory
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(project_root)

        # Configuration for different requirements files
        configs = [
            {
                "file": "requirements.txt",
                "extras": ["cpu", "full"],
                "description": "CPU version with full dependencies"
            },
            {
                "file": "requirements-cu118.txt",
                "extras": ["cu118", "full"],
                "description": "CUDA 11.8 version with full dependencies"
            },
            {
                "file": "requirements-cu121.txt",
                "extras": ["cu121", "full"],
                "description": "CUDA 12.1 version with full dependencies"
            },
            {
                "file": "requirements-cu124.txt",
                "extras": ["cu124", "full"],
                "description": "CUDA 12.4 version with full dependencies"
            },
            {
                "file": "requirements-cu126.txt",
                "extras": ["cu126", "full"],
                "description": "CUDA 12.6 version with full dependencies"
            },
            {
                "file": "requirements-cu128.txt",
                "extras": ["cu128", "full"],
                "description": "CUDA 12.8 version with full dependencies"
            },
        ]

        print("Generating requirements files...")
        print("=" * 50)

        success_count = 0
        total_count = len(configs)

        for config in configs:
            cmd = [
                "uv", "pip", "compile", "pyproject.toml",
                "-o", config["file"],
                "--no-deps"
            ]

            # Add extra flags for each extra dependency
            for extra in config["extras"]:
                cmd.extend(["--extra", extra])

            print(f"Generating {config['file']} ({config['description']})")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Success: {config['file']}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed: {config['file']} - {e}")
            except FileNotFoundError:
                print("Error: uv command not found")
                return False

            print("-" * 30)

        print(f"\nSummary: {success_count}/{total_count} requirements files generated successfully")

        if success_count == total_count:
            print("All requirements files generated successfully!")
            return True
        else:
            print(f"{total_count - success_count} files failed to generate")
            return False

    finally:
        # Change back to original directory
        os.chdir(original_cwd)


def main():
    """Main function."""
    print("Requirements File Generator")
    print("=" * 50)

    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv is not available. Please install uv first:")
        print("   pip install uv")
        sys.exit(1)

    # Generate requirements files
    success = generate_requirements_files()

    if not success:
        sys.exit(1)

    print("\nRequirements generation completed.")

if __name__ == "__main__":
    main()