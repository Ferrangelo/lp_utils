import copy
import json
import os
import re
import socket
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

SPEED_OF_LIGHT = 299_792.458  # km/s


def set_paths():
    host = socket.gethostname()

    if host.lower() == "fedorat14":
        ssd_lp_path = "/run/media/anferrar/ssd1tb/work_archives/lp/"
    elif ("fisso" in host.lower()) or ("pop" in host.lower()):
        ssd_lp_path = "/media/anferrar/ssd1tb/work_archives/lp/"
    elif "recas" in host.lower():
        ssd_lp_path = "/lustre/euclid/aferrar/lp/"
    else:
        ssd_lp_path = "/g100_scratch/userexternal/aferrar3/lp/"

    raygal_catalogs_path = ssd_lp_path + "catalogs/raygal/"
    raygal_diluted_path = raygal_catalogs_path + "diluted/"
    raygal_random_path = raygal_catalogs_path + "randoms/"
    raygal_test_path = raygal_catalogs_path + "test/"

    return (
        raygal_catalogs_path,
        raygal_diluted_path,
        raygal_random_path,
        raygal_test_path,
    )


def read_json(filename):
    # Get the package's config directory
    config_dir = Path(__file__).parent / "config"
    cosmo_file = config_dir / filename

    with open(cosmo_file, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    return json_dict


def parse_corrfunc_filename(filename):
    # Match patterns for different parameters in the filename
    pair_type_pattern = r"py_mocks_(\w+)smu"  # DDsmu, DRsmu, or RRsmu
    n_pattern = r"_(\d+\.\d+e[\+|-]\d+)"  # matches scientific notation numbers
    mumax_pattern = r"mumax_(\d+\.\d+)"
    mubins_pattern = r"mubins_(\d+)"
    smin_pattern = r"smin_(\d+\.\d+)"
    smax_pattern = r"smax_(\d+\.\d+)"
    sbinsize_pattern = r"sbinsize_(\d+\.\d+)"
    zmin_pattern = r"zmin_(\d+\.\d+)"
    zmax_pattern = r"zmax_(\d+\.\d+)"
    redshift_pattern = r"(?:_|^)(z(?:[0-5]|rsd))(?:\.dat)"  # matches z0-z5 or zrsd

    # Extract values using regex
    pair_type_match = re.search(pair_type_pattern, filename)
    numbers = re.findall(
        n_pattern, filename
    )  # This will find all scientific notation numbers
    mumax_match = re.search(mumax_pattern, filename)
    mubins_match = re.search(mubins_pattern, filename)
    smin_match = re.search(smin_pattern, filename)
    smax_match = re.search(smax_pattern, filename)
    sbinsize_match = re.search(sbinsize_pattern, filename)
    zmin_match = re.search(zmin_pattern, filename)
    zmax_match = re.search(zmax_pattern, filename)
    redshift_match = re.search(redshift_pattern, filename)

    # For DR files, we need both N and rand_N
    n_dict = {}
    if pair_type_match and pair_type_match.group(1) == "DR":
        n_dict = (
            {"N": float(numbers[0]), "rand_N": float(numbers[1])}
            if len(numbers) >= 2
            else {}
        )
    elif pair_type_match and pair_type_match.group(1) == "RR":
        n_dict = {"rand_N": float(numbers[0])} if numbers else {}
    else:
        n_dict = {"N": float(numbers[0])} if numbers else {}

    final_dict = {
        "pair_type": pair_type_match.group(1) if pair_type_match else None,
        **n_dict,
        "mumax": float(mumax_match.group(1)) if mumax_match else None,
        "mubins": int(mubins_match.group(1)) if mubins_match else None,
        "smin": float(smin_match.group(1)) if smin_match else None,
        "smax": float(smax_match.group(1)) if smax_match else None,
        "sbinsize": float(sbinsize_match.group(1)) if sbinsize_match else None,
        "zmin": float(zmin_match.group(1)) if zmin_match else None,
        "zmax": float(zmax_match.group(1)) if zmax_match else None,
        "redshift_key": redshift_match.group(1) if redshift_match else None,
    }

    return final_dict


def create_corrfunc_identifier(
    pair_type: str = None,
    N: float = None,
    rand_N: float = None,
    mumax: float = None,
    mubins: int = None,
    smin: float = None,
    smax: float = None,
    sbinsize: float = None,
    zmin: float = None,
    zmax: float = None,
    redshift_key: str = None,
) -> str:
    """
    Create a unique identifier based on the provided parameters.
    Only includes parameters that are not None.
    """
    parts = []

    if pair_type is not None:
        parts.append(pair_type)
    if N is not None:
        parts.append(f"N{N:.2e}")
    # if rand_N is not None and pair_type == "DR":
    if rand_N is not None:
        parts.append(f"Nr{rand_N:.2e}")
    if mumax is not None:
        parts.append(f"mumax{mumax}")
    if mubins is not None:
        parts.append(f"mubins{mubins}")
    if smin is not None:
        parts.append(f"smin{smin}")
    if smax is not None:
        parts.append(f"smax{smax}")
    if sbinsize is not None:
        parts.append(f"sbin{sbinsize}")
    if zmin is not None:
        parts.append(f"zmin{zmin}")
    if zmax is not None:
        parts.append(f"zmax{zmax}")
    if redshift_key is not None:
        parts.append(redshift_key)

    return "_".join(parts)


def find_corrfunc_files(
    directory: str,
    pair_type: str = None,
    N: float = None,
    rand_N: float = None,
    mumax: float = None,
    mubins: int = None,
    smin: float = None,
    smax: float = None,
    sbinsize: float = None,
    zmin: float = None,
    zmax: float = None,
    redshift_key: str = None,
    pattern: str = None,
    debug: bool = False,
) -> tuple[list, str]:
    """
    Find all Corrfunc output files matching the given criteria in the specified directory.
    Returns a tuple containing the list of matching files and a unique identifier.

    Parameters:
    -----------
    [... same as before ...]

    Returns:
    --------
    tuple
        (List of full paths to matching files, unique identifier string)
    """
    matching_files = []

    for filename in os.listdir(directory):
        if pattern and not re.search(pattern, filename):
            continue

        info = parse_corrfunc_filename(filename)
        if not info:
            continue

        matches = True

        params_to_check = {
            "pair_type": pair_type,
            "N": N,
            "rand_N": rand_N,
            "mumax": mumax,
            "mubins": mubins,
            "smin": smin,
            "smax": smax,
            "sbinsize": sbinsize,
            "zmin": zmin,
            "zmax": zmax,
            "redshift_key": redshift_key,
        }

        for param, expected in params_to_check.items():
            if expected is not None and info.get(param) != expected:
                if debug:
                    print(
                        f"{param} mismatch: expected {expected}, got {info.get(param)}"
                    )
                matches = False
        if matches:
            matching_files.append(os.path.join(directory, filename))

    # Create unique identifier based on input parameters
    identifier = create_corrfunc_identifier(
        pair_type=pair_type,
        N=N,
        rand_N=rand_N,
        mumax=mumax,
        mubins=mubins,
        smin=smin,
        smax=smax,
        sbinsize=sbinsize,
        zmin=zmin,
        zmax=zmax,
        redshift_key=redshift_key,
    )

    return matching_files, identifier


def extract_z_range(filename):
    zmin_match = re.search(r"zmin_(\d+\.\d+)", filename)
    zmax_match = re.search(r"zmax_(\d+\.\d+)", filename)

    zmin = float(zmin_match.group(1)) if zmin_match else None
    zmax = float(zmax_match.group(1)) if zmax_match else None

    return zmin, zmax

def _extract_x_key(filename):
    """Extract key like 'x10', 'x20' from filename."""
    match = re.search(r'x(\d+)', filename)
    return f"x{match.group(1)}" if match else None


def load_jsons_from_dir(json_dir, pattern=None, key_extractor=None):
    """
    Load JSON file paths into a dictionary with dynamic keys.

    Parameters:
    -----------
    json_dir : str
        Directory containing JSON files
    pattern : str, optional
        Regex pattern to filter files (default: matches files with x followed by numbers)
    key_extractor : callable, optional
        Function to extract key from filename (default: extracts 'x10', 'x20', etc.)

    Returns:
    --------
    dict : Dictionary with extracted keys and full file paths
    """

    # Default pattern to match files like 'narrow_z0_0.9_1.1_x10.json'
    if pattern is None:
        pattern = r".*x(\d+)\.json$"

    # Default key extractor to get 'x10', 'x20', etc.
    if key_extractor is None:
        key_extractor = _extract_x_key

    jsons = {}
    json_dir = Path(json_dir)

    for file_path in json_dir.glob("*.json"):
        filename = file_path.name

        # Check if file matches pattern
        if re.match(pattern, filename):
            key = key_extractor(filename)
            if key:
                jsons[key] = str(file_path)

    return jsons
