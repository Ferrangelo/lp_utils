import socket
import numpy as np
import polars as pl
from pathlib import Path
import json

SPEED_OF_LIGHT = 299_792.458  # km/s


def set_paths():
    host = socket.gethostname()

    if host.lower() == "fedorat14":
        ssd_lp_path = "/run/media/anferrar/ssd1tb/work_archives/lp/"
    else:
        ssd_lp_path = "/media/anferrar/ssd1tb/work_archives/lp/"

    raygal_catalogs_path = ssd_lp_path + "catalogs/raygal/"
    raygal_diluted_path = raygal_catalogs_path + "diluted/"
    raygal_random_path = raygal_catalogs_path + "randoms/"
    raygal_test_path = raygal_catalogs_path + "test/"

    return raygal_catalogs_path, raygal_diluted_path, raygal_random_path, raygal_test_path


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


def read_json(filename):
    # Get the package's config directory
    config_dir = Path(__file__).parent / "config"
    cosmo_file = config_dir / filename

    with open(cosmo_file, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    return json_dict
