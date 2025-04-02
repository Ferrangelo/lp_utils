import json
import os
import re
import socket
from pathlib import Path

import numpy as np
import polars as pl

SPEED_OF_LIGHT = 299_792.458  # km/s


def set_paths():
    host = socket.gethostname()

    if host.lower() == "fedorat14":
        ssd_lp_path = "/run/media/anferrar/ssd1tb/work_archives/lp/"
    elif ("fisso" in host.lower()) or ("pop" in host.lower()):
        ssd_lp_path = "/media/anferrar/ssd1tb/work_archives/lp/"
    else:
        ssd_lp_path = "/lustre/euclid/aferrar/lp/"

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


def corrfunc_angles_phi_neg(df, angle1_key, angle2_key):
    """
    Only when there are negative phi angles (like in the case of the narrow raygal lightcone)
    Transforms the input dataframe's angle columns to the format required by Corrfunc. Corrfunc expects right ascension (RA) in degrees from 0 to 360 and declination (DEC) in degrees from -90 to 90. This function converts the input angles, which are in the raygal coordinate system, to the Corrfunc-compatible RA and DEC format.
    Corrfunc wants RA in degrees from  0 to 360.
    Corrfunc wants DEC in degrees from  -90 to 90.
    The raygal angles (and therefore the randoms) spans beta1 from -7.5 to 57.5 and beta2 from 90 to 140 (pre rotation).
    To correct for this I transform radians in degrees and then I convert from my angles to RA and DEC using

    RA = beta_1 [deg] + 180 and
    DEC = 90 - beta_2 [deg].

    Previously I was doing like corrfunc does (it should change nothing):
    RA = beta_1 [deg] + 180 and DEC = beta_2 [deg]
    """

    df = df.with_columns(
        pl.col(angle1_key) * 180 / np.pi + 180, -pl.col(angle2_key) * 180 / np.pi + 90
    )
    return df


def filter_catalog(
    df,
    narrow=False,
    filter_z=False,
    angle1min=None,
    angle2min=None,
    width=None,
    height=None,
    zmin=None,
    zmax=None,
):
    if not narrow:
        return df.filter(pl.col("z0").is_between(0.05, 0.465))

    filters = filters_angles(df, angle1min, angle2min, width, height)

    if filter_z:
        if zmin is None or zmax is None:
            raise ValueError("zmin and zmax must be provided when filter_z is True")

        zcols = [col for col in df.columns if col.startswith("z")]
        if len(zcols) != 1:
            raise ValueError(f"Expected exactly one z column, found {len(zcols)}")

        zkey = zcols[0]
        print(f"zkey: {zkey}")
        filters.append(pl.col(zkey).is_between(zmin, zmax))

    return df.filter(pl.all_horizontal(filters))


def filters_angles(df, angle1min=None, angle2min=None, width=None, height=None):
    rect_params = read_json("params_rectangle_angles.json")

    angle1min = angle1min if angle1min is not None else rect_params["b1min"]
    angle2min = angle2min if angle2min is not None else rect_params["b2min"]
    width = width if width is not None else rect_params["width"]
    height = height if height is not None else rect_params["height"]

    angle1key = df.collect_schema().names()[0]  # ok for both lazy and dataframes
    angle2key = df.collect_schema().names()[1]
    print(angle1key, angle2key)
    filters = [
        pl.col(angle1key).is_between(angle1min, angle1min + width),
        pl.col(angle2key).is_between(angle2min, angle2min + height),
    ]
    return filters


def parse_catalog_filename(filename):
    # Match the exact string format between 'catalog_' and '_narrow'
    number_pattern = r"catalog_([0-9.e+]+)"
    zmin_pattern = r"zmin_(\d+\.\d+)"
    zmax_pattern = r"zmax_(\d+\.\d+)"
    redshift_pattern = r"(?:_|^)(z(?:[0-5]|rsd))(?:_|$)"  # matches z0-z5 or zrsd

    number_match = re.search(number_pattern, filename)
    zmin_match = re.search(zmin_pattern, filename)
    zmax_match = re.search(zmax_pattern, filename)
    redshift_match = re.search(redshift_pattern, filename)

    return {
        "catalog_number": number_match.group(1)
        if number_match
        else None,  # Keep as string
        "zmin": float(zmin_match.group(1)) if zmin_match else None,
        "zmax": float(zmax_match.group(1)) if zmax_match else None,
        "redshift_key": redshift_match.group(1) if redshift_match else None,
    }


def find_catalog_file(
    directory: str, zmin: float, zmax: float, redshift_key: str
) -> str:
    """
    Find a catalog file matching the given criteria in the specified directory.
    Returns the full path of the first matching file or None if no match is found.

    # Example usage:
    # matching_file = find_catalog_file(raygal_diluted_path, 0.9, 1.1, "z0")
    # if matching_file:
    #     print(f"Found matching file: {matching_file}")
    # else:
    #     print("No matching file found")

    """
    for filename in os.listdir(directory):
        info = parse_catalog_filename(filename)
        print(info)
        if (
            info
            and info["zmin"] == zmin
            and info["zmax"] == zmax
            and info["redshift_key"] == redshift_key
        ):
            return os.path.join(directory, filename)
    return None


def find_catalog_files(
    directory: str,
    zmin: float = None,
    zmax: float = None,
    redshift_key: str = None,
    pattern: str = None,
) -> list:
    """
    Find all catalog files matching the given criteria in the specified directory.
    Returns a list of full paths of matching files.

    # Example usage:
    # matching_files = find_catalog_files(
    #     raygal_diluted_path,
    #     zmin=0.9,
    #     zmax=1.1,
    #     redshift_key="z0",
    #     pattern="particles"  # optional pattern to pre-filter files
    # )

    """
    matching_files = []

    for filename in os.listdir(directory):
        if pattern and not re.search(pattern, filename):
            continue

        info = parse_catalog_filename(filename)
        if not info:
            continue

        matches = True
        if zmin is not None and info["zmin"] != zmin:
            matches = False
        if zmax is not None and info["zmax"] != zmax:
            matches = False
        if redshift_key is not None and info["redshift_key"] != redshift_key:
            matches = False

        if matches:
            matching_files.append(os.path.join(directory, filename))

    return matching_files


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
    elif pair_type_match and  pair_type_match.group(1) == "RR":
        n_dict = {"rand_N": float(numbers[0])} if numbers else {}
    else:
        n_dict = {"N": float(numbers[0])} if numbers else {}

    return {
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
    if rand_N is not None and pair_type == "DR":
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
        if pair_type is not None and info["pair_type"] != pair_type:
            matches = False
        if N is not None and info.get("N") != N:
            matches = False
        if rand_N is not None and info.get("rand_N") != rand_N:
            matches = False
        if mumax is not None and info["mumax"] != mumax:
            matches = False
        if mubins is not None and info["mubins"] != mubins:
            matches = False
        if smin is not None and info["smin"] != smin:
            matches = False
        if smax is not None and info["smax"] != smax:
            matches = False
        if sbinsize is not None and info["sbinsize"] != sbinsize:
            matches = False
        if zmin is not None and info["zmin"] != zmin:
            matches = False
        if zmax is not None and info["zmax"] != zmax:
            matches = False
        if redshift_key is not None and info["redshift_key"] != redshift_key:
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