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
    zkey=None,
):
    if not narrow:
        return df.filter(pl.col("z0").is_between(0.05, 0.465))

    filters = filters_angles(df, angle1min, angle2min, width, height)

    if filter_z:
        if zmin is None or zmax is None:
            raise ValueError("zmin and zmax must be provided when filter_z is True")

        zcols = [col for col in df.columns if col.startswith("z")]
        if not zcols:
            raise ValueError("No z column found")

        if zkey is None:
            zkey = zcols[0] if len(zcols) == 1 else "z0"
            msg = (
                f"Only one z column found: using {zkey}"
                if len(zcols) == 1
                else f"Multiple z columns found, using {zkey}"
            )
            print(msg)
        else:
            if zkey not in zcols:
                raise ValueError(
                    f"Specified zkey '{zkey}' not found in available z columns: {zcols}"
                )

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


def read_test_file_and_plot(filepath):
    samp_df = pl.read_parquet(filepath)
    b1 = None
    b1 = None
    z = None

    for col1, col2 in [("angle1", "angle2"), ("RA", "DEC"), ("beta1", "beta2")]:
        if col1 in samp_df.columns and col2 in samp_df.columns:
            b1 = samp_df[col1]
            b2 = samp_df[col2]
            break

    if b1 is not None:
        plt.figure(figsize=(3, 3))
        plt.scatter(b1, b2, s=0.1)
        plt.title("Raygal angles distribution (rotated)")
        plt.xlabel(r"angle1", fontsize=9)
        plt.ylabel(r"angle1", fontsize=9)
        plt.tight_layout()

    for col3 in ["d_or_z", "z0", "z1", "z2", "z3", "z4", "z5", "zrsd"]:
        if col3 in samp_df.columns:
            z = samp_df[col3]
            plt.figure(figsize=(8, 5))
            plt.hist(z, bins=50, alpha=0.5, label="Distances or z", edgecolor="black")
            plt.title(f"Histogram of {len(z)} Random Points")
            plt.xlabel("Values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()


def rotate_catalog(df_catalog, angle1_key, angle2_key):
    b1 = copy.deepcopy(df_catalog[angle1_key])
    b2 = copy.deepcopy(df_catalog[angle2_key])

    x = np.array(np.cos(b1) * np.sin(b2))
    y = np.array(np.sin(b1) * np.sin(b2))
    z = np.array(np.cos(b2))

    # Create rotation object
    # Create rotations and combine them
    Ry = Rotation.from_euler("y", -25, degrees=True)
    Rz = Rotation.from_euler("z", -25, degrees=True)
    Rtot = Ry * Rz

    rotated_coords = Rtot.apply(
        np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    )
    x_rot = rotated_coords[:, 0].reshape(x.shape)
    y_rot = rotated_coords[:, 1].reshape(y.shape)
    z_rot = rotated_coords[:, 2].reshape(z.shape)

    del x, y, z

    b1r = np.arctan(y_rot / x_rot)
    b2r = np.arccos(z_rot)

    df_catalog = df_catalog.with_columns(
        [pl.Series(name=angle1_key, values=b1r), pl.Series(name=angle2_key, values=b2r)]
    )

    return df_catalog


def get_output_filename(type, narrow, filter_z, zmin, zmax, N_particles, suffix):
    if not narrow:
        if filter_z:
            filename = f"{type}_catalog_{N_particles:.2e}_fullsky_zmin_{zmin}_zmax_{zmax}{suffix}"
    else:
        if filter_z:
            filename = f"{type}_catalog_{N_particles:.2e}_narrow_zmin_{zmin}_zmax_{zmax}{suffix}"
        else:
            filename = f"{type}_catalog_{N_particles:.2e}_narrow{suffix}"
    return filename


def extract_z_range(filename):
    zmin_match = re.search(r"zmin_(\d+\.\d+)", filename)
    zmax_match = re.search(r"zmax_(\d+\.\d+)", filename)

    zmin = float(zmin_match.group(1)) if zmin_match else None
    zmax = float(zmax_match.group(1)) if zmax_match else None

    return zmin, zmax


def get_file_length(filename: str) -> int:
    if filename.endswith(".parquet"):
        return pl.scan_parquet(filename).select(pl.len()).collect().item()
    elif filename.endswith(".txt"):
        return (
            pl.scan_csv(filename, separator=" ", has_header=False)
            .select(pl.len())
            .collect()
            .item()
        )
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def write_df_final_output(df_final, output_path, test_output_path):
    print(f"Writing output in {output_path}")
    (
        df_final.write_csv(
            output_path,
            separator=" ",
            include_header=False,
        )
    )

    print(f"Writing test output in {test_output_path}")
    df_final.sample(n=100000, with_replacement=False).write_parquet(test_output_path)


def check_correct_filtering(
    narrow,
    filtered_catalog,
    angle1_min,
    angle1_max,
    angle2_min,
    angle2_max,
    angle1_key,
    angle2_key,
    filter_z,
    zmin,
    zmax,
    redshift_key,
):
    if narrow:
        assert filtered_catalog[angle1_key].max() < angle1_max
        assert filtered_catalog[angle1_key].min() > angle1_min

        assert filtered_catalog[angle2_key].max() < angle2_max
        assert filtered_catalog[angle2_key].min() > angle2_min

    if filter_z:
        assert filtered_catalog[redshift_key].max() < zmax
        assert filtered_catalog[redshift_key].min() > zmin


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
        pattern = r'.*x(\d+)\.json$'
    
    # Default key extractor to get 'x10', 'x20', etc.
    if key_extractor is None:
        key_extractor = lambda filename: f"x{re.search(r'x(\d+)', filename).group(1)}" if re.search(r'x(\d+)', filename) else None
    
    jsons = {}
    json_dir = Path(json_dir)
    
    for file_path in json_dir.glob('*.json'):
        filename = file_path.name
        
        # Check if file matches pattern
        if re.match(pattern, filename):
            key = key_extractor(filename)
            if key:
                jsons[key] = str(file_path)
    
    return jsons