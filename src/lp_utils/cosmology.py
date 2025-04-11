import json
from pathlib import Path

import numpy as np
from scipy import integrate, interpolate
from scipy.special import hyp2f1
from scipy.optimize import fsolve

from lp_utils.utils import SPEED_OF_LIGHT, read_json


class Cosmology:
    def __init__(
        self,
        preset=None,
        Omega_r=None,
        Omega_m=None,
        Omega_DE=None,
        Omega_k=None,
        h=None,
        sigma8=None,
    ):
        if preset is not None:
            self.cosmo = self.choose_cosmo(preset)
            self.h = self.cosmo["h"]
            self.Omega_r = self.cosmo["Omega_r"]
            self.Omega_m = self.cosmo["Omega_m"]
            self.Omega_L = self.cosmo["Omega_DE"]
            self.Omega_DE = self.cosmo["Omega_DE"]
            self.Omega_k = self.cosmo["Omega_k"]
            self.sigma8 = self.cosmo["sigma8"]
            self.n_s = self.cosmo["n_s"]
            self.w = self.cosmo.get("w", -1.0)
            self.As = self.cosmo.get("As")
        else:
            if None in (Omega_r, Omega_m, Omega_DE, Omega_k, h, sigma8):
                raise ValueError(
                    "Must provide all cosmological parameters or use a preset"
                )
            self.Omega_r = Omega_r
            self.Omega_m = Omega_m
            self.Omega_DE = Omega_DE
            self.Omega_L = Omega_DE
            self.Omega_k = Omega_k
            self.sigma8 = sigma8

    def choose_cosmo(self, cosmology):
        list_of_cosmologies = [
            "raygal",
            "istf",
            "wmap1",
            "wmap3",
            "wmap5",
            "wmap7",
            "wmap9",
        ]
        if cosmology.lower() in list_of_cosmologies:
            cosmo_dict = self._read_cosmo_file(f"{cosmology}_cosmology.json")
        else:
            raise ValueError(
                f"Unknown preset cosmology: {cosmology}. Available presets are: {', '.join(list_of_cosmologies)}"
            )
        return cosmo_dict

    def _read_cosmo_file(self, filename):
        cosmo_dict = read_json(filename)

        # For istf, calculate Omega_r if not provided
        if "istf" in filename and "Omega_r" not in cosmo_dict:
            cosmo_dict["Omega_r"] = (
                1.0
                - cosmo_dict["Omega_m"]
                - cosmo_dict["Omega_k"]
                - cosmo_dict["Omega_DE"]
            )

        return cosmo_dict

    def E_late_times(self, z):
        return np.sqrt(
            self.Omega_m * (1 + z) ** 3 + self.Omega_k * (1 + z) ** 2 + self.Omega_L
        )

    def E_correct(self, z):
        return np.sqrt(
            self.Omega_r * (1 + z) ** 4
            + self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_L
        )

    def comoving_distance(self, z, h_units=True):
        h = 1.0
        if not h_units:
            h = self.h
        Dh = SPEED_OF_LIGHT * 0.01 / h
        integral = integrate.quad(
            lambda x: 1.0 / self.E_correct(x), 0.0, z, points=10000
        )
        return Dh * integral[0]

    def comoving_distance_late_times(self, z, h_units=True):
        h = 1.0
        if not h_units:
            h = self.h
        Dh = SPEED_OF_LIGHT * 0.01 / h
        integral = integrate.quad(
            lambda x: 1.0 / self.E_late_times(x), 0.0, z, points=10000
        )
        return Dh * integral[0]

    def comoving_distance_interp(self, use_late_times=False):
        z_vals = np.linspace(0.0, 2.5, 4000)
        distance_func = (
            self.comoving_distance_late_times
            if use_late_times
            else self.comoving_distance
        )
        dist_vals = np.array([distance_func(z) for z in z_vals])
        distance_cubic_interp = interpolate.interp1d(z_vals, dist_vals, kind="cubic")
        return distance_cubic_interp

    def growth_factor(self, z_input, norm="0"):
        z = np.asarray(z_input)
        a = 1 / (1 + z)
        if norm == "MD":
            return hyp2f1(1 / 3, 1, 11 / 6, (self.Omega_m - 1) * a**3 / (self.Omega_m))
        elif norm == "0":
            return (
                a
                * hyp2f1(1 / 3, 1, 11 / 6, (self.Omega_m - 1) * a**3 / self.Omega_m)
                / hyp2f1(1 / 3, 1, 11 / 6, (self.Omega_m - 1) / self.Omega_m)
            )
        else:
            raise ValueError("unknown norm type")

    def volume_zbin(self, zi, zf, fsky=None, solid_angle=None, use_late_times=False):
        r = self.comoving_distance_interp(use_late_times)
        if fsky is not None:
            omega = 4 * np.pi * fsky
        elif solid_angle is not None:
            omega = solid_angle
        else:
            raise ValueError("Either fsky or solid_angle must be provided")
        return omega * (r(zf) ** 3 - r(zi) ** 3) / 3

    def get_vol_interp(self, zmin, zmax, fsky=None, solid_angle=None):
        # Create interpolator for volume_zbin function
        z_grid = np.linspace(0, 2.5, 1000)  # Create fine z grid
        if solid_angle is not None:
            fsky = solid_angle / (4 * np.pi)
        vol_grid = np.array(
            [self.volume_zbin(zmin, zmax, fsky) for z in z_grid]
        )  # Calculate volumes
        volume_interp = interpolate.interp1d(z_grid, vol_grid, kind="cubic")
        return volume_interp

    def find_z_for_target_volume(self, volume_target, fsky, z_min=0, z_max=2):
        return fsolve(
            lambda z: self.volume_zbin(z, z_max, fsky=fsky) - volume_target,
            (z_max + z_min) / 2,
        )[0]


def growth_factor(z, Om=0.31, norm="0"):
    z = np.asarray(z)
    a = 1 / (1 + z)
    if norm == "MD":
        return hyp2f1(1 / 3, 1, 11 / 6, (Om - 1) * a**3 / (Om))
    elif norm == "0":
        return (
            a
            * hyp2f1(1 / 3, 1, 11 / 6, (Om - 1) * a**3 / Om)
            / hyp2f1(1 / 3, 1, 11 / 6, (Om - 1) / Om)
        )
    else:
        raise ValueError("unknown norm type")


def xiLS(N, Nr, dd_of_s, dr_of_s, rr_of_s):
    return (
        Nr * (Nr - 1) / (N * (N - 1)) * dd_of_s - 2 * (Nr - 1) / N * dr_of_s
    ) / rr_of_s + 1


def change_sigma8(k, P, sigma8_wanted):
    def filt(q, R):
        return 3.0 * (np.sin(q * R) - q * R * np.cos(q * R)) / (q * R) ** 3

    integrand = P / (2.0 * np.pi) ** 3 * filt(k, 8) ** 2 * k**2
    sigma8_old = np.sqrt(4.0 * np.pi * integrate.simpson(integrand, k))

    new_P = P * (sigma8_wanted / sigma8_old) ** 2

    sigma8_computed = np.sqrt(
        4.0
        * np.pi
        * integrate.simpson(new_P / (2 * np.pi) ** 3 * filt(k, 8) ** 2 * k**2, k)
    )

    np.isclose(sigma8_computed, sigma8_wanted)

    return new_P


def bacco_params(cosmo_dict, expfactor=1):
    bacco_dict = {
        "omega_cold": cosmo_dict["Omega_m"],
        "sigma8_cold": cosmo_dict["sigma8"],
        "omega_baryon": cosmo_dict["Omega_b"],
        "ns": cosmo_dict["n_s"],
        "hubble": cosmo_dict["h"],
        "neutrino_mass": cosmo_dict.get("m_nu", 0),
        "w0": cosmo_dict.get("w0", -1.0),
        "wa": cosmo_dict.get("wa", 0.0),
        "expfactor": expfactor,
    }

    return bacco_dict
