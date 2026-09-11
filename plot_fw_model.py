#!/usr/bin/env python3
"""Plot FASTWIND line profiles and model diagnostics from the command line.

Run from the FASTWIND work directory (the directory containing FORMAL_INPUT):

    python plot_fw_model.py MODEL [R] [VSINI] [SNR]

Examples:

    python plot_fw_model.py T40g37R15N60
    python plot_fw_model.py T40g37R15N60 7500
    python plot_fw_model.py T40g37R15N60 7500 100
    python plot_fw_model.py T40g37R15N60 7500 100 150
    python plot_fw_model.py T40g37R15N60 7500 100 150 --seed 12345

MODEL can be a subdirectory of the current work directory or a full path.
INDAT.DAT is always read from MODEL; FORMAL_INPUT is read from the work
(current) directory. Commented ':T' FORMAL_INPUT line definitions are retained
for component markers, and OUT files without any matching definition are still
plotted without markers.

Two PDFs are written to MODEL:

    <model>_line_profiles.pdf                     # if R=VSINI=SNR=0
    <model>_line_profiles_<settings>.pdf          # otherwise
    <model>_model_diagnostics.pdf                 # model diagnostics

Only one of the two line-profile filename forms is produced in a given run.

The plotted line profile starts from FASTWIND's intrinsic PROF column, not its
PROFROT column. The optional synthetic-observation processing is:

    intrinsic -> rotational broadening -> instrumental broadening
              -> realistic detector sampling -> noise

Rotational broadening uses PyAstronomy.pyasl.rotBroad with a linear
limb-darkening coefficient of 0.6. Instrumental broadening uses Astropy's
Gaussian1DKernel/convolve at constant R=lambda/Delta(lambda_FWHM). If R>0, the
final spectrum is sampled at three pixels per resolution element.

SNR is the continuum S/N per displayed observational pixel. The noise model is
photon dominated, with a small fixed background/read-noise variance term. Thus
absorption cores have smaller *absolute* noise but lower relative S/N, while
emission peaks have larger absolute noise. By default every run uses a new
random realization. Use --seed to reproduce one exactly.

Matplotlib's non-interactive Agg backend is forced before pyplot is imported,
which also prevents Qt/Wayland backend warnings in this batch PDF script.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import secrets
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Must be set before pyplot (or any plotting dependency) is imported.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    from astropy.convolution import Gaussian1DKernel, convolve
except ImportError:
    Gaussian1DKernel = None
    convolve = None

try:
    from PyAstronomy import pyasl
except ImportError:
    pyasl = None


# Plot/synthetic-observation constants. Model, R, vsini, S/N, and seed are CLI.
N_COLUMNS = 3
LINE_FIG_WIDTH_INCH = 8.27
PANEL_HEIGHT_INCH = 2.15
MODEL_BOX_HEIGHT_INCH = 2.05
TITLE_HEIGHT_INCH = 0.42
SUBTITLE_HEIGHT_INCH = 0.30
DIAGNOSTICS_FIGSIZE = (11.69, 9.25)

LIMB_DARKENING = 0.60
PIXELS_PER_RESOLUTION_ELEMENT = 3.0
NOISE_BACKGROUND_FRACTION = 0.02
C_KMS = 299792.458

@dataclass
class ModelParameters:
    name: str
    teff: float
    logg: float
    radius_rsun: float
    rmax_rstar: float
    tmin_teff: float
    mdot: float
    vmin_kms: float
    vinf_kms: float
    beta: float
    vtrans_vsound: float
    nhe: float
    xihe: float
    vturb_kms: float
    metallicity_zsun: float
    abundances: Dict[str, float]
    clumping_mode: str
    clumping_params: Dict[str, float]

    @property
    def log_luminosity_lsun(self) -> float:
        # IAU nominal solar values / CODATA constants, in cgs.
        r_sun_cm = 6.957e10
        sigma_sb_cgs = 5.670374419e-5  # erg cm^-2 s^-1 K^-4
        l_sun_cgs = 3.828e33           # erg s^-1
        radius_cm = self.radius_rsun * r_sun_cm
        luminosity = 4.0 * np.pi * radius_cm**2 * sigma_sb_cgs * self.teff**4
        return float(np.log10(luminosity / l_sun_cgs))

    @property
    def spectroscopic_mass_msun(self) -> float:
        # log g in FASTWIND is cgs.
        g_cgs = 10.0 ** self.logg
        r_sun_cm = 6.957e10
        g_const_cgs = 6.67430e-8
        m_sun_g = 1.98847e33
        r_cm = self.radius_rsun * r_sun_cm
        mass_g = g_cgs * r_cm**2 / g_const_cgs
        return float(mass_g / m_sun_g)

    @property
    def wind_strength_q(self) -> float:
        """FASTWIND optical-depth invariant Q in conventional units.

        Q = Mdot / (R_star * v_inf)^(3/2), with Mdot in Msun/yr,
        R_star in Rsun, and v_inf in km/s.
        """
        if self.radius_rsun <= 0 or self.vinf_kms <= 0:
            return float("nan")
        return float(self.mdot / (self.radius_rsun * self.vinf_kms) ** 1.5)

    @property
    def log_wind_strength_q(self) -> float:
        q = self.wind_strength_q
        if not np.isfinite(q) or q <= 0:
            return float("nan")
        return float(np.log10(q))


@dataclass(frozen=True)
class Component:
    lower: str
    upper: str
    line_number: int
    stark_option: int


@dataclass
class FormalLine:
    name: str
    components: List[Component]


@dataclass
class ComponentMarker:
    wavelength: float
    label: str
    transition_key: Tuple[str, str]


@dataclass
class Profile:
    wavelength: np.ndarray
    flux: np.ndarray
    equivalent_width: Optional[float]


@dataclass
class ProfileEntry:
    line_name: str
    suffix: str
    path: Path
    formal_line: Optional[FormalLine]


@dataclass
class TempStructure:
    radius_inner: np.ndarray
    temperature: np.ndarray
    ne_lte: Optional[np.ndarray]
    r_tau23: Optional[float]


@dataclass
class ModelStructure:
    teff: float
    ggrav: float
    radius_cm: float
    yhe: float
    mu: float
    vmax_cms: float
    mdot_gs: float
    beta: float
    radius_inner: np.ndarray
    velocity_fraction: np.ndarray
    dvdr_scaled: np.ndarray
    density: np.ndarray
    ne: np.ndarray
    nh: np.ndarray
    clumping: np.ndarray
    index: np.ndarray
    pressure: np.ndarray
    ns: Optional[int]
    updated: Optional[bool]
    xt: Optional[float]


@dataclass
class FluxContData:
    wavelength: np.ndarray
    log_fnu: np.ndarray
    trad: np.ndarray
    trad1: np.ndarray
    rthin: np.ndarray
    lthin: np.ndarray
    rtau1: np.ndarray
    r_tau23: Optional[float]
    flux_radius_inner: np.ndarray
    flux_logtau: np.ndarray
    flux_error: np.ndarray


def _fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _as_bool(text: str) -> bool:
    value = text.strip().upper()
    if value in {"T", ".TRUE.", "TRUE"}:
        return True
    if value in {"F", ".FALSE.", "FALSE"}:
        return False
    raise ValueError(f"Cannot interpret {text!r} as a Fortran logical")


def _clean_data_lines(path: Path) -> List[str]:
    """Return active, non-empty lines, stripping FASTWIND ':T' comments."""
    out: List[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(":T", 1)[0].strip()
        if line:
            out.append(line)
    return out


def parse_indat(path: Path) -> ModelParameters:
    """Parse stellar, wind, abundance, and simple-clumping INDAT inputs."""
    lines = _clean_data_lines(path)
    if len(lines) < 11:
        raise ValueError(f"{path} is too short to be a FASTWIND INDAT.DAT")

    name = lines[0].strip().strip("'\"").strip()

    teff, logg, radius = map(_fortran_float, lines[3].split()[:3])
    rmax, tmin = map(_fortran_float, lines[4].split()[:2])
    mdot, vmin, vinf, beta, vtrans = map(_fortran_float, lines[5].split()[:5])
    nhe, xihe = map(_fortran_float, lines[6].split()[:2])

    opt_line = lines[7].split()
    if len(opt_line) < 2:
        raise ValueError(f"Cannot parse FASTWIND logical options in {path}")
    optlucy = _as_bool(opt_line[1])

    metal_line = lines[8].split()
    vturb = _fortran_float(metal_line[0])
    metallicity = abs(_fortran_float(metal_line[1]))

    # After the fixed parameter block comes either the ordinary microclumping
    # line CLF,VCLSTART,VCLMAX, or THICK plus four macroclumping-related lines.
    i = 10
    clumping_mode = "unknown"
    clumping_params: Dict[str, float] = {}
    if i < len(lines):
        tok = lines[i].split()
        if tok and tok[0].upper() == "THICK":
            clumping_mode = "thick"
            # Current FASTWIND versions use THICK + four parameter lines.
            i += 1
            thick_lines = lines[i:i + 4]
            i += min(4, len(thick_lines))
            # Keep a few numeric values if available, but do not assume a
            # specific macroclumping parameterization in the plot header.
            numeric: List[float] = []
            for row in thick_lines:
                for item in row.split():
                    try:
                        numeric.append(_fortran_float(item))
                    except ValueError:
                        pass
            if numeric:
                clumping_params["first"] = numeric[0]
        else:
            clumping_mode = "micro"
            try:
                clf, vclstart, vclmax = map(_fortran_float, tok[:3])
                clumping_params = {
                    "clf_max": clf,
                    "vcl_start": vclstart,
                    "vcl_max": vclmax,
                }
            except (ValueError, IndexError):
                pass
            i += 1

    # If the modified-Lucy temperature option is off, optional Hopf input
    # follows the clumping block before any abundance overrides.
    if not optlucy and i < len(lines):
        try:
            hopfself = _as_bool(lines[i].split()[0])
        except ValueError:
            hopfself = True
        else:
            i += 1
            if not hopfself and i < len(lines):
                i += 1

    abundances: Dict[str, float] = {}
    for line in lines[i:]:
        tok = line.split()
        if len(tok) < 2:
            continue
        element = tok[0].upper()
        if element == "XRAYS":
            break
        try:
            value = _fortran_float(tok[1])
        except ValueError:
            continue
        abundances[element] = value

    return ModelParameters(
        name=name,
        teff=teff,
        logg=logg,
        radius_rsun=radius,
        rmax_rstar=rmax,
        tmin_teff=tmin,
        mdot=mdot,
        vmin_kms=vmin,
        vinf_kms=vinf,
        beta=beta,
        vtrans_vsound=vtrans,
        nhe=nhe,
        xihe=xihe,
        vturb_kms=vturb,
        metallicity_zsun=metallicity,
        abundances=abundances,
        clumping_mode=clumping_mode,
        clumping_params=clumping_params,
    )


def _formal_content_line(raw: str) -> Tuple[str, bool]:
    """Return FORMAL_INPUT content and whether the whole physical line was commented.

    A leading ':T' is removed so that old/commented line definitions remain
    available for OUT files from earlier models. A second/inline ':T' still
    starts a true comment and is discarded.
    """
    line = raw.strip()
    if not line:
        return "", False
    commented = line.startswith(":T")
    if commented:
        line = line[2:].strip()
    if ":T" in line:
        line = line.split(":T", 1)[0].strip()
    return line, commented


def parse_formal_input_all(path: Path) -> Tuple[float, List[FormalLine]]:
    """Parse VSINI and every line definition, active or commented out.

    FASTWIND line definitions can wrap over several physical lines. This
    parser treats a commented leading ':T' as disabled-but-readable input,
    enabling component markers for profiles that already exist in MODEL_DIR.
    """
    raw_lines = path.read_text(errors="replace").splitlines()

    # VSINI is the first active standalone numeric input line.
    vsini = 0.0
    for raw in raw_lines:
        text, commented = _formal_content_line(raw)
        if commented or not text:
            continue
        tok = text.split()
        if len(tok) != 1:
            continue
        try:
            vsini = _fortran_float(tok[0])
            break
        except ValueError:
            pass

    result: List[FormalLine] = []
    current_name: Optional[str] = None
    current_ncomp = 0
    component_tokens: List[str] = []

    def finalize_current() -> None:
        nonlocal current_name, current_ncomp, component_tokens
        if current_name is None:
            return
        needed = 4 * current_ncomp
        if len(component_tokens) < needed:
            warnings.warn(
                f"Incomplete FORMAL_INPUT definition for {current_name}: "
                f"needed {needed} component tokens, found {len(component_tokens)}"
            )
        else:
            components: List[Component] = []
            ok = True
            for k in range(current_ncomp):
                group = component_tokens[4 * k:4 * k + 4]
                try:
                    components.append(
                        Component(group[0], group[1], int(group[2]), int(group[3]))
                    )
                except (ValueError, IndexError):
                    ok = False
                    break
            if ok:
                result.append(FormalLine(current_name, components))
        current_name = None
        current_ncomp = 0
        component_tokens = []

    for raw in raw_lines:
        text, _commented = _formal_content_line(raw)
        if not text:
            continue
        tok = text.split()
        if not tok:
            continue

        # A new definition has a nonnumeric name followed by integer NCOMP.
        is_new = False
        if len(tok) >= 2:
            try:
                ncomp = int(tok[1])
            except ValueError:
                ncomp = -1
            else:
                try:
                    _fortran_float(tok[0])
                    first_is_numeric = True
                except ValueError:
                    first_is_numeric = False
                is_new = (not first_is_numeric and ncomp >= 0)

        if is_new:
            finalize_current()
            current_name = tok[0]
            current_ncomp = ncomp
            component_tokens = tok[2:]
        elif current_name is not None and len(component_tokens) < 4 * current_ncomp:
            component_tokens.extend(tok)

        if current_name is not None and len(component_tokens) >= 4 * current_ncomp:
            finalize_current()

    finalize_current()
    return vsini, result


def find_atom_file(run_dir: Path, inicalc_dir: Path) -> Path:
    candidates = [run_dir / "ATOM_FILE", inicalc_dir / "ATOM_FILE"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find ATOM_FILE. Checked: " + ", ".join(str(x) for x in candidates)
    )


def _resolve_fastwind_data_file(
    filename: str, run_dir: Path, inicalc_dir: Path
) -> Optional[Path]:
    candidates = [
        run_dir / filename,
        inicalc_dir / filename,
        inicalc_dir / "DATA" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_atom_file_references(
    atom_file_path: Path, run_dir: Path, inicalc_dir: Path
) -> Tuple[Path, Optional[Path]]:
    entries = [
        line.strip()
        for line in atom_file_path.read_text(errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith(":T")
    ]
    if not entries:
        raise ValueError(f"{atom_file_path} is empty")

    detail_name = entries[0].split()[0]
    lines_name = entries[2].split()[0] if len(entries) >= 3 else None

    detail_path = _resolve_fastwind_data_file(detail_name, run_dir, inicalc_dir)
    if detail_path is None:
        raise FileNotFoundError(
            f"ATOM_FILE requests {detail_name!r}, but it was not found in RUN_DIR, "
            "INICALC_DIR, or INICALC_DIR/DATA"
        )

    lines_path = None
    if lines_name:
        lines_path = _resolve_fastwind_data_file(lines_name, run_dir, inicalc_dir)
        if lines_path is None:
            warnings.warn(
                f"Packed-component line table {lines_name!r} listed in ATOM_FILE "
                "was not found; those component markers may be unavailable."
            )

    return detail_path, lines_path


def parse_detail_level_frequencies(path: Path) -> Dict[str, float]:
    """Read level frequencies (Hz) from DETAIL atom-file L blocks."""
    frequencies: Dict[str, float] = {}
    in_level_block = False

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(":T", 1)[0].strip()
        if not line:
            continue
        tok = line.split()

        if len(tok) == 1 and tok[0].upper() == "L":
            in_level_block = True
            continue

        if in_level_block and tok[0] == "0":
            in_level_block = False
            continue

        if not in_level_block or len(tok) < 4:
            continue

        try:
            _ = _fortran_float(tok[1])
            freq = _fortran_float(tok[2])
        except ValueError:
            continue
        frequencies[tok[0].upper()] = freq

    return frequencies


def parse_packed_component_wavelengths(
    path: Optional[Path], needed: Iterable[Tuple[str, str, int]]
) -> Dict[Tuple[str, str, int], float]:
    """Read exact packed-component wavelengths from FASTWIND LINES_*.dat."""
    needed_upper = {(a.upper(), b.upper(), n) for a, b, n in needed if n != 0}
    found: Dict[Tuple[str, str, int], float] = {}
    if path is None or not needed_upper:
        return found

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(":T", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        if len(tok) < 4:
            continue
        try:
            n = int(tok[2])
            wave = _fortran_float(tok[3])
        except ValueError:
            continue
        key = (tok[0].upper(), tok[1].upper(), n)
        if key in needed_upper:
            found[key] = wave

    return found


def fastwind_air_wavelength_from_frequencies(freq1: float, freq2: float) -> float:
    """Reproduce FASTWIND PREFORMAL/HALLCL1 wavelength calculation."""
    c_angstrom_s = 2.997925e18  # exact value used in FASTWIND HALLCL1
    df = abs(freq1 - freq2)
    if df == 0:
        raise ValueError("Identical DETAIL level frequencies")
    wave = c_angstrom_s / df

    # FASTWIND's two historical He I corrections, applied before refraction.
    if abs(wave - 4026.72472212120) < 1e-10:
        wave = 4027.3
    if abs(wave - 4388.82518552703) < 1e-10:
        wave = 4389.16

    if wave <= 2000.0:  # UV wavelengths are kept in vacuum.
        return wave

    xkw = 1.0e4 / wave
    n_air = 1.0 + 1.0e-7 * (
        643.28 + 294981.0 / (146.0 - xkw**2) + 2554.0 / (41.0 - xkw**2)
    )
    return wave / n_air


def _roman(n: int) -> str:
    values = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, numeral in values:
        while n >= value:
            out += numeral
            n -= value
    return out


def _species_from_line_name(line_name: str) -> Optional[str]:
    m = re.match(r"^(HE|SI|[HCNOP])(I{1,5}|IV|V)", line_name.upper())
    if not m:
        return None
    element = {"HE": "He", "SI": "Si"}.get(m.group(1), m.group(1).title())
    return f"{element} {m.group(2)}"


def _decode_level(level: str, line_name: str) -> Tuple[Optional[str], str]:
    """Convert FASTWIND labels such as N38 / HE212 / SI404 to readable form."""
    raw = level.upper()
    patterns = [
        (r"^(HE)(\d)(.+)$", "He"),
        (r"^(SI)(\d)(.+)$", "Si"),
        (r"^([HCNOP])(\d)(.+)$", None),
    ]
    for pattern, fixed_element in patterns:
        m = re.match(pattern, raw)
        if not m:
            continue
        token = m.group(1)
        ion = int(m.group(2))
        suffix = m.group(3).lstrip("_")
        if suffix.isdigit():
            suffix = str(int(suffix))
        element = fixed_element or token.title()
        return f"{element} {_roman(ion)}", suffix
    return _species_from_line_name(line_name), level


def component_transition_label(component: Component, line_name: str) -> str:
    species_l, lower = _decode_level(component.lower, line_name)
    species_u, upper = _decode_level(component.upper, line_name)
    species = species_l or species_u or "component"
    return f"{species} {lower}\u2192{upper}"


def get_component_markers(
    formal_line: FormalLine,
    level_frequencies: Dict[str, float],
    packed_wavelengths: Dict[Tuple[str, str, int], float],
) -> List[ComponentMarker]:
    markers: List[ComponentMarker] = []

    for component in formal_line.components:
        lo = component.lower.upper()
        up = component.upper.upper()
        key = (lo, up, component.line_number)
        wave: Optional[float] = None

        if component.line_number != 0:
            wave = packed_wavelengths.get(key)
        else:
            f_lo = level_frequencies.get(lo)
            f_up = level_frequencies.get(up)
            if f_lo is not None and f_up is not None:
                try:
                    wave = fastwind_air_wavelength_from_frequencies(f_lo, f_up)
                except ValueError:
                    wave = None

        if wave is None:
            warnings.warn(
                f"Could not determine wavelength for {formal_line.name}: "
                f"{component.lower} {component.upper} line {component.line_number}"
            )
            continue

        markers.append(
            ComponentMarker(
                wavelength=wave,
                label=component_transition_label(component, formal_line.name),
                transition_key=(component.lower.upper(), component.upper.upper()),
            )
        )

    return markers


def read_out_profile(path: Path) -> Profile:
    """Read FASTWIND OUT.<LINE>, deliberately using intrinsic PROF."""
    wavelength: List[float] = []
    flux: List[float] = []
    equivalent_width: Optional[float] = None

    for raw in path.read_text(errors="replace").splitlines():
        tok = raw.split()
        if not tok:
            continue
        if len(tok) >= 6:
            try:
                _ = int(tok[0])
                values = [_fortran_float(x) for x in tok[1:6]]
            except ValueError:
                continue
            wavelength.append(values[1])  # wavelength [A]
            flux.append(values[3])        # PROF, intrinsic normalized profile
            continue
        if len(tok) == 1 and wavelength:
            try:
                equivalent_width = _fortran_float(tok[0])
            except ValueError:
                pass

    if len(wavelength) < 3:
        raise ValueError(f"No FASTWIND profile table found in {path}")

    wave = np.asarray(wavelength, dtype=float)
    prof = np.asarray(flux, dtype=float)
    good = np.isfinite(wave) & np.isfinite(prof) & (wave > 0)
    wave = wave[good]
    prof = prof[good]
    order = np.argsort(wave)
    wave = wave[order]
    prof = prof[order]
    keep = np.concatenate(([True], np.diff(wave) > 0))
    return Profile(wave[keep], prof[keep], equivalent_width)


def _processing_grid(
    wavelength: np.ndarray,
    resolving_power: float,
    vsini: float,
) -> np.ndarray:
    """Uniform wavelength grid fine enough for the requested convolutions."""
    wave = np.asarray(wavelength, dtype=float)
    if len(wave) < 3 or (resolving_power <= 0 and vsini <= 0):
        return np.array(wave, copy=True)

    native_step = float(np.median(np.diff(wave)))
    lambda_mid = float(np.sqrt(wave[0] * wave[-1]))
    target_steps = [native_step]
    if resolving_power > 0:
        target_steps.append(lambda_mid / resolving_power / 10.0)
    if vsini > 0:
        target_steps.append(lambda_mid * vsini / C_KMS / 12.0)

    # Avoid pathological memory use while retaining a very well sampled kernel.
    min_allowed = (wave[-1] - wave[0]) / 100000.0
    step = max(min(target_steps), min_allowed)
    n = max(3, int(math.ceil((wave[-1] - wave[0]) / step)) + 1)
    n = min(n, 100001)
    return np.linspace(wave[0], wave[-1], n)


def rotational_broaden(
    wavelength: np.ndarray,
    flux: np.ndarray,
    vsini: float,
    epsilon: float = LIMB_DARKENING,
) -> np.ndarray:
    """Apply Gray rotational broadening with PyAstronomy.pyasl.rotBroad."""
    if vsini is None or vsini <= 0:
        return np.array(flux, copy=True)
    if pyasl is None:
        raise ImportError(
            "PyAstronomy is required for rotational broadening. Install it "
            "with 'pip install PyAstronomy', or use VSINI=0."
        )
    if len(wavelength) < 5:
        return np.array(flux, copy=True)
    return np.asarray(
        pyasl.rotBroad(
            np.asarray(wavelength, dtype=float),
            np.asarray(flux, dtype=float),
            float(epsilon), float(vsini), edgeHandling="firstlast",
        ),
        dtype=float,
    )


def instrumental_broaden(
    wavelength: np.ndarray,
    flux: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    """Gaussian instrumental broadening at constant resolving power."""
    if resolving_power is None or resolving_power <= 0:
        return np.array(flux, copy=True)
    if Gaussian1DKernel is None or convolve is None:
        raise ImportError(
            "Astropy is required for instrumental broadening. Install astropy "
            "or use resolving power R=0."
        )
    if len(wavelength) < 5:
        return np.array(flux, copy=True)

    ln_wave = np.log(wavelength)
    dln = np.median(np.diff(ln_wave))
    if not np.isfinite(dln) or dln <= 0:
        raise ValueError("Wavelength grid must be strictly increasing")
    ln_uniform = np.arange(ln_wave[0], ln_wave[-1] + 0.5*dln, dln)
    flux_uniform = np.interp(ln_uniform, ln_wave, flux)

    fwhm_ln_lambda = 1.0 / resolving_power
    sigma_ln_lambda = fwhm_ln_lambda / (2.0*np.sqrt(2.0*np.log(2.0)))
    sigma_pix = sigma_ln_lambda / dln
    if sigma_pix < 1e-3:
        return np.array(flux, copy=True)

    kernel = Gaussian1DKernel(stddev=sigma_pix)
    broadened = convolve(
        flux_uniform, kernel, boundary="extend", normalize_kernel=True,
        nan_treatment="interpolate",
    )
    return np.interp(ln_wave, ln_uniform, broadened)


def _observation_sampling_grid(
    wavelength: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    """Sample an R-broadened spectrum at three pixels per resolution element."""
    if resolving_power is None or resolving_power <= 0:
        return np.array(wavelength, copy=True)
    dln = 1.0 / (resolving_power * PIXELS_PER_RESOLUTION_ELEMENT)
    ln0 = math.log(float(wavelength[0]))
    ln1 = math.log(float(wavelength[-1]))
    n = max(3, int(math.floor((ln1-ln0)/dln)) + 1)
    ln_grid = ln0 + np.arange(n, dtype=float)*dln
    if ln_grid[-1] < ln1 - 0.25*dln:
        ln_grid = np.append(ln_grid, ln1)
    return np.exp(ln_grid)


def add_synthetic_noise(
    flux: np.ndarray,
    continuum_snr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add photon-dominated noise for a requested continuum S/N.

    With normalized local flux f, background variance fraction b, and continuum
    S/N S, sigma(f) = sqrt((max(f,0)+b)/(1+b))/S.  This preserves exactly the
    requested continuum S/N and gives realistic flux-dependent shot noise.
    """
    if continuum_snr is None or continuum_snr <= 0:
        return np.array(flux, copy=True)
    f = np.asarray(flux, dtype=float)
    b = float(NOISE_BACKGROUND_FRACTION)
    variance_shape = (np.clip(f, 0.0, None) + b) / (1.0 + b)
    sigma = np.sqrt(variance_shape) / continuum_snr
    return f + rng.normal(0.0, sigma, size=f.shape)


def simulate_observation(
    wavelength: np.ndarray,
    intrinsic_flux: np.ndarray,
    resolving_power: float,
    vsini: float,
    continuum_snr: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intrinsic -> rotation -> instrumental broadening -> sampling -> noise."""
    wave = np.asarray(wavelength, dtype=float)
    flux = np.asarray(intrinsic_flux, dtype=float)

    if resolving_power <= 0 and vsini <= 0:
        proc_wave = np.array(wave, copy=True)
        proc_flux = np.array(flux, copy=True)
    else:
        proc_wave = _processing_grid(wave, resolving_power, vsini)
        proc_flux = np.interp(proc_wave, wave, flux)
        proc_flux = rotational_broaden(proc_wave, proc_flux, vsini)
        proc_flux = instrumental_broaden(proc_wave, proc_flux, resolving_power)

    if resolving_power > 0:
        obs_wave = _observation_sampling_grid(wave, resolving_power)
        obs_flux = np.interp(obs_wave, proc_wave, proc_flux)
    else:
        obs_wave = np.array(wave, copy=True)
        obs_flux = np.interp(obs_wave, proc_wave, proc_flux)

    return obs_wave, add_synthetic_noise(obs_flux, continuum_snr, rng)


def resolve_model_dir(run_dir: Path, model_dir: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir.expanduser().resolve()
    return (run_dir / model_dir).expanduser().resolve()


def _split_profile_filename(path: Path) -> Tuple[str, str]:
    """Return (line name, broadening suffix) from OUT.<LINE>[_VTxxx]."""
    if not path.name.startswith("OUT."):
        raise ValueError(path.name)
    body = path.name[4:]
    m = re.match(
        r"^(?P<line>.+?)(?P<suffix>_(?:VTV?|VT)\d+|_ESC(?:_(?:VTV?|VT)\d+)?)?$",
        body,
        flags=re.IGNORECASE,
    )
    if not m:
        return body, ""
    return m.group("line"), (m.group("suffix") or "")


def discover_profiles(
    model_dir: Path,
    formal_lines: Sequence[FormalLine],
) -> List[ProfileEntry]:
    """Discover every OUT file, whether or not the line is active in FORMAL_INPUT."""
    formal_by_upper = {line.name.upper(): line for line in formal_lines}
    order = {line.name.upper(): i for i, line in enumerate(formal_lines)}
    selected: List[ProfileEntry] = []
    for path in sorted(model_dir.glob("OUT.*")):
        if not path.is_file():
            continue
        line_name, suffix = _split_profile_filename(path)
        selected.append(ProfileEntry(
            line_name=line_name,
            suffix=suffix,
            path=path,
            formal_line=formal_by_upper.get(line_name.upper()),
        ))
    selected.sort(key=lambda e: (
        order.get(e.line_name.upper(), 10**9), e.line_name.upper(), e.suffix.upper()
    ))
    return selected


def _format_mdot(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2g}\times10^{{{exponent}}}"


def _epsilon_abundance(input_value: float) -> float:
    """Convert either FASTWIND abundance convention to 12+log10(X/H)."""
    return input_value if input_value > 0 else 12.0 + input_value


def _model_parameter_columns(params: ModelParameters) -> List[Tuple[str, List[str]]]:
    stellar = [
        rf"$T_{{\rm eff}}$ = {params.teff:,.0f} K",
        rf"$\log g$ = {params.logg:.2f}",
        rf"$R_\star$ = {params.radius_rsun:g} $R_\odot$",
        rf"$R_{{\rm max}}$ = {params.rmax_rstar:g} $R_\star$",
        rf"$T_{{\rm min}}/T_{{\rm eff}}$ = {params.tmin_teff:g}",
    ]
    wind = [
        rf"$\dot{{M}}$ = ${_format_mdot(params.mdot)}$ $M_\odot$ yr$^{{-1}}$",
        rf"$v_\infty$ = {params.vinf_kms:g} km s$^{{-1}}$",
        rf"$\beta$ = {params.beta:g}",
        rf"$v_{{\rm min}}$ = {params.vmin_kms:g} km s$^{{-1}}$",
        rf"$v_{{\rm trans}}$ = {params.vtrans_vsound:g} $a_{{\rm s}}$",
    ]
    atmosphere = [
        rf"$v_{{\rm turb,model}}$ = {params.vturb_kms:g} km s$^{{-1}}$",
        rf"$\xi_{{\rm He}}$ = {params.xihe:g}",
    ]
    if params.clumping_mode == "micro" and params.clumping_params:
        cp = params.clumping_params
        atmosphere.extend([
            rf"$f_{{\rm cl,max}}$ = {cp.get('clf_max', float('nan')):g}",
            rf"$v_{{\rm cl,start}}/v_\infty$ = {cp.get('vcl_start', float('nan')):g}",
            rf"$v_{{\rm cl,max}}/v_\infty$ = {cp.get('vcl_max', float('nan')):g}",
        ])
    elif params.clumping_mode == "thick":
        atmosphere.append("macroclumping: THICK")
    return [("Stellar", stellar), ("Wind", wind), ("Atmosphere", atmosphere)]


def _draw_model_parameter_box(ax: plt.Axes, params: ModelParameters) -> None:
    """Draw the large model-parameter box, including derived quantities."""
    ax.set_axis_off()
    box = FancyBboxPatch(
        (0.018, 0.035), 0.964, 0.925,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=ax.transAxes, facecolor="white", edgecolor="0.58", linewidth=0.9,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(0.035, 0.900, "Stellar & wind parameters", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.0, fontweight="bold", clip_on=True)

    # The same internal x positions are used for the upper three groups and the
    # lower derived row, so related quantities line up cleanly.
    xcols = [0.045, 0.365, 0.695]
    for x, (heading, lines) in zip(xcols, _model_parameter_columns(params)):
        ax.text(x, 0.785, heading, transform=ax.transAxes, ha="left", va="center",
                fontsize=7.8, fontweight="bold", color="0.25", clip_on=True)
        y = 0.680
        for line in lines:
            ax.text(x, y, line, transform=ax.transAxes, ha="left", va="center",
                    fontsize=7.05, clip_on=True)
            y -= 0.094

    # Dividers between Stellar / Wind / Atmosphere, then a full-width divider
    # above the derived quantities.
    for x in (0.345, 0.675):
        ax.plot([x, x], [0.260, 0.825], transform=ax.transAxes,
                color="0.86", linewidth=0.7, clip_on=True)
    ax.plot([0.035, 0.965], [0.220, 0.220], transform=ax.transAxes,
            color="0.86", linewidth=0.7, clip_on=True)

    ax.text(0.045, 0.165, "Derived parameters", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.8, fontweight="bold",
            color="0.25", clip_on=True)
    log_q = params.log_wind_strength_q
    q_text = (rf"$\log Q$ = {log_q:.3f}"
              if np.isfinite(log_q) else r"$\log Q$ = n/a")
    derived = [
        rf"$\log(L/L_\odot)$ = {params.log_luminosity_lsun:.3f}",
        rf"$M_{{\rm spec}}$ = {params.spectroscopic_mass_msun:.2f} $M_\odot$",
        q_text,
    ]
    for x, line in zip(xcols, derived):
        ax.text(x, 0.085, line, transform=ax.transAxes,
                ha="left", va="center", fontsize=7.25, clip_on=True)


def _draw_abundance_box(ax: plt.Axes, params: ModelParameters) -> None:
    ax.set_axis_off()
    box = FancyBboxPatch(
        (0.025, 0.035), 0.950, 0.925,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=ax.transAxes, facecolor="white", edgecolor="0.58", linewidth=0.9,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(0.055, 0.900, "Abundances", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.0, fontweight="bold", clip_on=True)
    ax.plot([0.50, 0.50], [0.125, 0.820], transform=ax.transAxes,
            color="0.86", linewidth=0.7, clip_on=True)
    ax.text(0.065, 0.775, "Global", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.7, fontweight="bold", color="0.25", clip_on=True)
    ax.text(0.065, 0.655, rf"$Z$ = {params.metallicity_zsun:g} $Z_\odot$",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.25, clip_on=True)
    ax.text(0.065, 0.545, rf"$N_{{\rm He}}$ = {params.nhe:g}",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.25, clip_on=True)

    ax.text(0.555, 0.775, "Individual", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.7, fontweight="bold", color="0.25", clip_on=True)
    if params.abundances:
        items = list(params.abundances.items())
        n = len(items)
        y_top, y_bottom = 0.655, 0.160
        ys = [0.655] if n == 1 else np.linspace(y_top, y_bottom, n)
        fontsize = max(5.25, 7.15 - 0.22 * max(0, n - 4))
        for y, (element, value) in zip(ys, items):
            ax.text(0.555, float(y),
                    rf"$\epsilon_{{\rm {element.title()}}}$ = {_epsilon_abundance(value):.2f}",
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=fontsize, clip_on=True)
    else:
        ax.text(0.555, 0.655, "-", transform=ax.transAxes,
                ha="left", va="center", fontsize=8.0, clip_on=True)


def _formal_vturb_values(profiles: Sequence[ProfileEntry]) -> List[float]:
    """Infer formal-integral vturb values from FASTWIND OUT filename suffixes."""
    values = set()
    for entry in profiles:
        # FASTWIND documentation: absent ending => vturb = 0 km/s.  Common
        # endings include _VT015, _VTV015, and _ESC_VT015.
        m = re.search(r"(?:^|_)VTV?(\d+)", entry.suffix, flags=re.IGNORECASE)
        values.add(float(int(m.group(1))) if m else 0.0)
    return sorted(values)


def _format_formal_vturb(values: Sequence[float]) -> str:
    if not values:
        return r"$v_{\rm turb,formal}$ = unknown"
    formatted = ", ".join(f"{v:g}" for v in values)
    unit = r"km s$^{-1}$"
    if len(values) == 1:
        return rf"$v_{{\rm turb,formal}}$ = {formatted} {unit}"
    return rf"$v_{{\rm turb,formal}}$ = {formatted} {unit}"


def _observation_subtitle(
    resolving_power: float,
    vsini: float,
    continuum_snr: float,
    noise_seed: Optional[int],
    formal_vturb_values: Sequence[float],
) -> str:
    rtext = rf"$R$ = {resolving_power:g}" if resolving_power > 0 else r"$R$ = intrinsic"
    vstext = rf"$v\sin i$ = {vsini:g} km s$^{{-1}}$"
    if continuum_snr > 0:
        seed_text = str(noise_seed) if noise_seed is not None else "unknown"
        sntext = rf"$S/N$ = {continuum_snr:g} (seed: {seed_text})"
    else:
        sntext = r"$S/N$ = none"
    vttext = _format_formal_vturb(formal_vturb_values)
    return "   |   ".join([rtext, vstext, sntext, vttext])


def _style_profile_axis(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=7.8, direction="in", top=True, right=True)
    ax.margins(x=0.015)


def make_line_profile_overview(
    run_dir: Path,
    inicalc_dir: Path,
    model_dir: Path,
    params: ModelParameters,
    resolving_power: float,
    vsini: float,
    continuum_snr: float,
    noise_seed: Optional[int],
    formal_lines: Sequence[FormalLine],
) -> Path:
    profiles = discover_profiles(model_dir, formal_lines)
    if not profiles:
        raise FileNotFoundError(f"No OUT.<LINE> profile files found in {model_dir}")

    formal_vturb = _formal_vturb_values(profiles)

    level_frequencies: Dict[str, float] = {}
    packed_wavelengths: Dict[Tuple[str, str, int], float] = {}
    detail_path: Optional[Path] = None
    lines_path: Optional[Path] = None
    try:
        atom_ref = find_atom_file(run_dir, inicalc_dir)
        detail_path, lines_path = read_atom_file_references(atom_ref, run_dir, inicalc_dir)
        level_frequencies = parse_detail_level_frequencies(detail_path)
        packed_needed = {
            (c.lower, c.upper, c.line_number)
            for entry in profiles if entry.formal_line is not None
            for c in entry.formal_line.components if c.line_number != 0
        }
        packed_wavelengths = parse_packed_component_wavelengths(lines_path, packed_needed)
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"Atomic component markers disabled: {exc}")

    n_panels = len(profiles)
    n_rows = int(math.ceil(n_panels / N_COLUMNS))
    fig_height = (TITLE_HEIGHT_INCH + SUBTITLE_HEIGHT_INCH + MODEL_BOX_HEIGHT_INCH
                  + PANEL_HEIGHT_INCH*n_rows + 0.30)
    fig = plt.figure(figsize=(LINE_FIG_WIDTH_INCH, fig_height))
    height_ratios = [MODEL_BOX_HEIGHT_INCH/PANEL_HEIGHT_INCH] + [1.0]*n_rows
    gs = fig.add_gridspec(
        n_rows+1, N_COLUMNS, height_ratios=height_ratios,
        left=0.075, right=0.985, bottom=0.045, top=0.925,
        wspace=0.30, hspace=0.40,
    )

    fig.suptitle(f"{params.name} - FASTWIND line profiles",
                 fontsize=16, fontweight="bold", y=0.988)
    fig.text(
        0.5, 0.955,
        _observation_subtitle(resolving_power, vsini, continuum_snr,
                              noise_seed, formal_vturb),
        ha="center", va="center", fontsize=10.3,
    )
    _draw_model_parameter_box(fig.add_subplot(gs[0, :2]), params)
    _draw_abundance_box(fig.add_subplot(gs[0, 2]), params)

    axes: List[plt.Axes] = [
        fig.add_subplot(gs[row+1, col])
        for row in range(n_rows) for col in range(N_COLUMNS)
    ]
    rng = np.random.default_rng(noise_seed)
    counts: Dict[str, int] = {}
    for entry in profiles:
        counts[entry.line_name.upper()] = counts.get(entry.line_name.upper(), 0) + 1

    for ax, entry in zip(axes, profiles):
        profile = read_out_profile(entry.path)
        obs_wave, obs_flux = simulate_observation(
            profile.wavelength, profile.flux, resolving_power, vsini,
            continuum_snr, rng,
        )

        markers: List[ComponentMarker] = []
        if entry.formal_line is not None and level_frequencies:
            markers = get_component_markers(entry.formal_line,
                                             level_frequencies, packed_wavelengths)
            markers = [m for m in markers if obs_wave[0] <= m.wavelength <= obs_wave[-1]]

        transition_colors: Dict[Tuple[str, str], object] = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for marker in markers:
            if marker.transition_key not in transition_colors:
                transition_colors[marker.transition_key] = color_cycle[
                    len(transition_colors) % len(color_cycle)]

        used_labels = set()
        for marker in markers:
            label = marker.label if marker.label not in used_labels else None
            ax.axvline(marker.wavelength, linestyle="--", linewidth=0.85,
                       alpha=0.72, color=transition_colors[marker.transition_key],
                       label=label, zorder=1)
            used_labels.add(marker.label)

        ax.plot(obs_wave, obs_flux,
                linewidth=(0.82 if continuum_snr > 0 else 1.20),
                color="black", zorder=3)
        ax.axhline(1.0, linestyle=":", linewidth=0.8, color="0.45", zorder=0)
        title = entry.line_name
        if counts.get(entry.line_name.upper(), 0) > 1 and entry.suffix:
            title += entry.suffix
        ax.set_title(title, fontsize=10.5, pad=4)
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=8.5)
        ax.set_ylabel(r"$F_\lambda/F_{\rm cont}$", fontsize=8.5)
        _style_profile_axis(ax)

        finite = obs_flux[np.isfinite(obs_flux)]
        if finite.size:
            ymin, ymax = min(np.nanmin(finite), 1.0), max(np.nanmax(finite), 1.0)
            span = max(ymax-ymin, 0.08)
            ax.set_ylim(ymin-0.10*span, ymax+0.18*span)
        if used_labels:
            ax.legend(loc="best", fontsize=6.3, frameon=False,
                      handlelength=1.8, handletextpad=0.4, borderaxespad=0.4)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    suffix_parts: List[str] = []
    if resolving_power > 0:
        suffix_parts.append(f"R{resolving_power:g}")
    if vsini > 0:
        suffix_parts.append(f"vsini{vsini:g}")
    if continuum_snr > 0:
        suffix_parts.append(f"SNR{continuum_snr:g}")
    settings_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_path = model_dir / f"{params.name}_line_profiles{settings_suffix}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Profiles plotted:   {n_panels}")
    print(f"Formal vturb:       {', '.join(f'{v:g}' for v in formal_vturb)} km/s")
    print(f"DETAIL atom file:   {detail_path if detail_path else 'not available'}")
    print(f"Packed line table:  {lines_path if lines_path else 'not available/not used'}")
    print(f"Line-profile PDF:   {output_path}")
    return output_path


# =============================================================================
# Model diagnostics
# =============================================================================


def _safe_loadtxt(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path.name}: {exc}")
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def read_tau_ross(path: Path) -> Optional[np.ndarray]:
    arr = _safe_loadtxt(path)
    if arr is None or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def read_fluxcont(path: Path) -> Optional[FluxContData]:
    if not path.exists():
        return None

    cont_rows: List[List[float]] = []
    flux_rows: List[List[float]] = []
    r_tau23: Optional[float] = None
    in_flux_block = False

    for raw in path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if "FLUX-ERROR" in stripped.upper():
            in_flux_block = True
            continue
        tok = stripped.split()

        if in_flux_block:
            if len(tok) >= 4:
                try:
                    row = [float(tok[0]), _fortran_float(tok[1]), _fortran_float(tok[2]), _fortran_float(tok[3])]
                except ValueError:
                    continue
                flux_rows.append(row)
            continue

        if len(tok) >= 8:
            try:
                _ = int(tok[0])
                row = [
                    _fortran_float(tok[1]), _fortran_float(tok[2]),
                    _fortran_float(tok[3]), _fortran_float(tok[4]),
                    _fortran_float(tok[5]), float(tok[6]), _fortran_float(tok[7]),
                ]
            except ValueError:
                continue
            cont_rows.append(row)
            continue

        # The scalar immediately after the continuum table is RTAU23.
        if cont_rows and len(tok) == 1 and r_tau23 is None:
            try:
                r_tau23 = _fortran_float(tok[0])
            except ValueError:
                pass

    if not cont_rows and not flux_rows:
        return None

    cont = np.asarray(cont_rows, dtype=float) if cont_rows else np.empty((0, 7))
    flux = np.asarray(flux_rows, dtype=float) if flux_rows else np.empty((0, 4))
    return FluxContData(
        wavelength=cont[:, 0] if len(cont) else np.array([]),
        log_fnu=cont[:, 1] if len(cont) else np.array([]),
        trad=cont[:, 2] if len(cont) else np.array([]),
        trad1=cont[:, 3] if len(cont) else np.array([]),
        rthin=cont[:, 4] if len(cont) else np.array([]),
        lthin=cont[:, 5] if len(cont) else np.array([]),
        rtau1=cont[:, 6] if len(cont) else np.array([]),
        r_tau23=r_tau23,
        flux_radius_inner=flux[:, 1] if len(flux) else np.array([]),
        flux_logtau=flux[:, 2] if len(flux) else np.array([]),
        flux_error=flux[:, 3] if len(flux) else np.array([]),
    )


def read_temp_structure(
    path: Path,
    nd_hint: Optional[int] = None,
    r_tau23_fallback: Optional[float] = None,
) -> Optional[TempStructure]:
    if not path.exists():
        return None
    vals = np.fromstring(path.read_text(errors="replace"), sep=" ")
    if vals.size < 4:
        return None

    nd: Optional[int] = nd_hint
    if nd is None:
        if (len(vals) - 1) % 3 == 0:
            nd = (len(vals) - 1) // 3
        elif len(vals) % 2 == 0:
            nd = len(vals) // 2
    if nd is None or nd <= 0 or len(vals) < 2 * nd:
        return None

    radius = vals[:nd].copy()
    temp = vals[nd:2 * nd].copy()
    ne: Optional[np.ndarray] = None
    r_tau23 = r_tau23_fallback
    if len(vals) >= 3 * nd:
        ne = vals[2 * nd:3 * nd].copy()
    if len(vals) >= 3 * nd + 1:
        r_tau23 = float(vals[3 * nd])

    return TempStructure(radius, temp, ne, r_tau23)


def read_model_binary(path: Path, nd_hint: Optional[int] = None) -> Optional[ModelStructure]:
    """Read current FASTWIND v10 MODEL sequential-unformatted record.

    The v10 payload is:
      8 doubles,
      R,V,GRADV,RHO,XNE,XNH,CLFAC (7*ND doubles),
      INDEX (ND int32), P (ND doubles), NS (int32), UPDATED (logical/int32), XT (double).
    """
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) < 16:
        return None

    endian: Optional[str] = None
    record_len = 0
    for candidate in ("<", ">"):
        n0 = struct.unpack(candidate + "I", raw[:4])[0]
        if n0 + 8 == len(raw) and struct.unpack(candidate + "I", raw[-4:])[0] == n0:
            endian = candidate
            record_len = n0
            break
    if endian is None:
        warnings.warn("MODEL record markers were not recognized; structural MODEL diagnostics skipped")
        return None

    if nd_hint is None:
        rem = record_len - 80
        if rem < 0 or rem % 68 != 0:
            warnings.warn("Could not infer FASTWIND depth count from MODEL record")
            return None
        nd = rem // 68
    else:
        nd = nd_hint
        if 80 + 68 * nd != record_len:
            warnings.warn(
                f"MODEL record size does not match ND={nd}; attempting to infer ND instead"
            )
            rem = record_len - 80
            if rem < 0 or rem % 68 != 0:
                return None
            nd = rem // 68

    payload = memoryview(raw)[4:4 + record_len]
    offset = 0
    f8 = np.dtype(endian + "f8")
    i4 = np.dtype(endian + "i4")

    params = np.frombuffer(payload, dtype=f8, count=8, offset=offset).astype(float)
    offset += 8 * 8

    arrays: List[np.ndarray] = []
    for _ in range(7):
        arrays.append(np.frombuffer(payload, dtype=f8, count=nd, offset=offset).astype(float).copy())
        offset += 8 * nd

    index = np.frombuffer(payload, dtype=i4, count=nd, offset=offset).astype(int).copy()
    offset += 4 * nd
    pressure = np.frombuffer(payload, dtype=f8, count=nd, offset=offset).astype(float).copy()
    offset += 8 * nd

    ns = int(np.frombuffer(payload, dtype=i4, count=1, offset=offset)[0])
    updated = bool(np.frombuffer(payload, dtype=i4, count=1, offset=offset + 4)[0])
    xt = float(np.frombuffer(payload, dtype=f8, count=1, offset=offset + 8)[0])

    return ModelStructure(
        teff=params[0], ggrav=params[1], radius_cm=params[2], yhe=params[3],
        mu=params[4], vmax_cms=params[5], mdot_gs=params[6], beta=params[7],
        radius_inner=arrays[0], velocity_fraction=arrays[1],
        dvdr_scaled=arrays[2], density=arrays[3], ne=arrays[4], nh=arrays[5],
        clumping=arrays[6], index=index, pressure=pressure,
        ns=ns, updated=updated, xt=xt,
    )


def read_clumping_output(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.genfromtxt(path, skip_header=1)
    except Exception as exc:
        warnings.warn(f"Could not read CLUMPING_OUTPUT: {exc}")
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 11:
        return None
    return arr


def read_grad_out(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    idx: List[int] = []
    grad: List[float] = []
    for raw in path.read_text(errors="replace").splitlines():
        tok = raw.split()
        if len(tok) < 3 or tok[0].upper() != "RADACC":
            continue
        try:
            idx.append(int(tok[1]))
            grad.append(_fortran_float(tok[2]))
        except ValueError:
            continue
    if not idx:
        return None
    return np.asarray(idx, dtype=int), np.asarray(grad, dtype=float)


def _combine_legends(ax: plt.Axes, ax2: Optional[plt.Axes] = None, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, **kwargs)


def _style_diag_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    ax.grid(False)


def make_model_diagnostics(
    run_dir: Path,
    model_dir: Path,
    params: ModelParameters,
) -> Optional[Path]:
    tau = read_tau_ross(model_dir / "TAU_ROS")
    nd_hint = len(tau) if tau is not None else None
    fluxcont = read_fluxcont(model_dir / "FLUXCONT")
    flux_rtau = fluxcont.r_tau23 if fluxcont is not None else None
    temp = read_temp_structure(model_dir / "TEMP", nd_hint=nd_hint, r_tau23_fallback=flux_rtau)
    model = read_model_binary(model_dir / "MODEL", nd_hint=nd_hint)
    clump = read_clumping_output(model_dir / "CLUMPING_OUTPUT")
    converg = _safe_loadtxt(model_dir / "CONVERG")
    converg_metals = _safe_loadtxt(model_dir / "CONVERG_METALS")
    max_tcorr = _safe_loadtxt(model_dir / "MAXTCORR.dat")
    grad = read_grad_out(model_dir / "GRAD.OUT")

    if all(x is None for x in (tau, temp, model, clump, fluxcont, converg, grad)):
        warnings.warn("No model-diagnostic files found; diagnostics PDF not created")
        return None

    r_tau23 = None
    if temp is not None and temp.r_tau23 is not None:
        r_tau23 = temp.r_tau23
    elif fluxcont is not None and fluxcont.r_tau23 is not None:
        r_tau23 = fluxcont.r_tau23

    fig = plt.figure(figsize=DIAGNOSTICS_FIGSIZE)
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.72, 1.0, 1.0],
        left=0.060, right=0.970, bottom=0.065, top=0.935,
        wspace=0.38, hspace=0.48,
    )
    fig.suptitle(f"{params.name} - FASTWIND model diagnostics",
                 fontsize=16, fontweight="bold", y=0.988)
    _draw_model_parameter_box(fig.add_subplot(gs[0, :2]), params)
    _draw_abundance_box(fig.add_subplot(gs[0, 2]), params)
    axes_flat = [fig.add_subplot(gs[row, col]) for row in (1, 2) for col in range(3)]

    # 1) Thermal structure + Rosseland optical depth.
    ax = axes_flat[0]
    if temp is not None:
        r = temp.radius_inner.copy()
        if r_tau23 is not None and r_tau23 > 0:
            r = r / r_tau23
        order = np.argsort(r)
        ax.plot(r[order], temp.temperature[order] / 1000.0, label=r"$T$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/R_\star$")
        ax.set_ylabel(r"$T$ [kK]")
        ax.axvline(1.0, linestyle=":", linewidth=0.8, color="0.45")
        ax2 = None
        if tau is not None and len(tau) == len(r):
            tau_aligned = tau[::-1, 0]
            good = tau_aligned > 0
            ax2 = ax.twinx()
            ax2.plot(r[good], np.log10(tau_aligned[good]), linestyle="--", label=r"$\log\tau_{\rm Ross}$")
            ax2.set_ylabel(r"$\log\tau_{\rm Ross}$")
            ax2.tick_params(direction="in", labelsize=8)
        _combine_legends(ax, ax2, loc="best", fontsize=7, frameon=False)
        ax.set_title("Thermal / optical-depth structure")
    else:
        ax.text(0.5, 0.5, "TEMP not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Thermal / optical-depth structure")
    _style_diag_axis(ax)

    # 2) Velocity + density structure.
    ax = axes_flat[1]
    if model is not None:
        r = model.radius_inner.copy()
        if r_tau23 is not None and r_tau23 > 0:
            r = r / r_tau23
        order = np.argsort(r)
        vel = model.velocity_fraction * model.vmax_cms * 1e-5
        ax.plot(r[order], vel[order], label=r"$v$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/R_\star$")
        ax.set_ylabel(r"$v$ [km s$^{-1}$]")
        ax.axvline(1.0, linestyle=":", linewidth=0.8, color="0.45")
        ax2 = ax.twinx()
        good = model.density > 0
        ax2.plot(r[good][np.argsort(r[good])], np.log10(model.density[good][np.argsort(r[good])]), linestyle="--", label=r"$\log\rho$")
        ax2.set_ylabel(r"$\log\rho$ [g cm$^{-3}$]")
        ax2.tick_params(direction="in", labelsize=8)
        _combine_legends(ax, ax2, loc="best", fontsize=7, frameon=False)
        ax.set_title("Velocity / density structure")
    else:
        ax.text(0.5, 0.5, "MODEL not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Velocity / density structure")
    _style_diag_axis(ax)

    # 3) Clumping stratification against v/v_infinity.
    ax = axes_flat[2]
    if clump is not None:
        vfrac = clump[:, 2]
        order = np.argsort(vfrac)
        ax.plot(vfrac[order], clump[:, 4][order], label=r"$f_{\rm cl}$")
        optional = [
            (5, r"$f_{\rm ic}$"),
            (6, r"$f_{\rm vel}$"),
            (8, r"$f_{\rm vol}$"),
        ]
        for col, label in optional:
            values = clump[:, col]
            if np.nanmax(np.abs(values - 1.0)) > 1e-4 or (np.nanmax(values) - np.nanmin(values)) > 1e-4:
                ax.plot(vfrac[order], values[order], label=label)
        ax.set_xlabel(r"$v/v_\infty$")
        ax.set_ylabel("Clumping quantities")
        ax.set_xlim(left=max(0.0, np.nanmin(vfrac) - 0.02), right=min(1.02, np.nanmax(vfrac) + 0.02))
        ax.legend(loc="best", fontsize=7, frameon=False)
        ax.set_title("Clumping stratification")
    else:
        ax.text(0.5, 0.5, "CLUMPING_OUTPUT not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Clumping stratification")
    _style_diag_axis(ax)

    # 4) Flux conservation error.
    ax = axes_flat[3]
    if fluxcont is not None and len(fluxcont.flux_error):
        r = fluxcont.flux_radius_inner.copy()
        if r_tau23 is not None and r_tau23 > 0:
            r = r / r_tau23
        order = np.argsort(r)
        ferr_pct = 100.0 * fluxcont.flux_error
        ax.plot(r[order], ferr_pct[order])
        ax.axhline(0.0, linestyle=":", linewidth=0.8, color="0.45")
        ax.axvline(1.0, linestyle=":", linewidth=0.8, color="0.45")
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/R_\star$")
        ax.set_ylabel(r"Flux error [%]")
        maxerr = np.nanmax(np.abs(ferr_pct))
        ax.set_title(rf"Flux conservation ($\max|\Delta F/F|={maxerr:.2f}\%$)")
    else:
        ax.text(0.5, 0.5, "FLUXCONT flux-error block not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Flux conservation")
    _style_diag_axis(ax)

    # 5) Convergence history; convert MEANERR = mean(log10 err) back to a
    # geometric-mean fractional error so all curves share a log y-axis.
    ax = axes_flat[4]
    plotted = False
    if converg is not None and converg.shape[1] >= 3:
        it = converg[:, 0]
        emax = converg[:, 1]
        mean_fraction = np.power(10.0, converg[:, 2])
        good = emax > 0
        ax.semilogy(it[good], emax[good], label="max NLTE correction")
        good = mean_fraction > 0
        ax.semilogy(it[good], mean_fraction[good], label="geometric mean NLTE error")
        plotted = True
    if converg_metals is not None and converg_metals.shape[1] >= 2:
        good = converg_metals[:, 1] > 0
        ax.semilogy(converg_metals[good, 0], converg_metals[good, 1], label="metal error")
        plotted = True
    if max_tcorr is not None and max_tcorr.shape[1] >= 3:
        good = max_tcorr[:, 1] > 0
        ax.semilogy(max_tcorr[good, 2], max_tcorr[good, 1], marker="o", markersize=2.5, linewidth=0.9, label="max T correction")
        plotted = True
    if plotted:
        ax.axhline(3e-3, linestyle=":", linewidth=0.8, color="0.45", label=r"$3\times10^{-3}$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fractional error / correction")
        ax.legend(loc="best", fontsize=6.5, frameon=False)
    else:
        ax.text(0.5, 0.5, "Convergence files not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Convergence history")
    _style_diag_axis(ax)

    # 6) Photospheric radiative acceleration relative to gravity.
    ax = axes_flat[5]
    if grad is not None and tau is not None:
        idx, gradacc = grad
        valid_idx = (idx >= 1) & (idx <= len(tau))
        idx0 = idx[valid_idx] - 1
        tau_here = tau[idx0, 0]
        ggrav = model.ggrav if model is not None else 10.0 ** params.logg
        good = tau_here > 0
        if np.any(good):
            ax.plot(np.log10(tau_here[good]), gradacc[valid_idx][good] / ggrav)
            ax.axhline(1.0, linestyle=":", linewidth=0.8, color="0.45")
            ax.set_xlabel(r"$\log\tau_{\rm Ross}$")
            ax.set_ylabel(r"$g_{\rm rad}/g$")
            ax.set_title("Radiative acceleration")
        else:
            ax.text(0.5, 0.5, "No positive Rosseland depths", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Radiative acceleration")
    else:
        ax.text(0.5, 0.5, "GRAD.OUT / TAU_ROS not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Radiative acceleration")
    _style_diag_axis(ax)


    output_path = model_dir / f"{params.name}_model_diagnostics.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Diagnostics PDF:    {output_path}")
    return output_path


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a FASTWIND model and optionally mimic a telescope observation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot_fw_model.py T40g37R15N60\n"
            "  python plot_fw_model.py T40g37R15N60 7500\n"
            "  python plot_fw_model.py T40g37R15N60 7500 100\n"
            "  python plot_fw_model.py T40g37R15N60 7500 100 150\n"
            "  python plot_fw_model.py T40g37R15N60 7500 100 150 --seed 12345"
        ),
    )
    parser.add_argument("model",
        help="Model directory relative to the current work directory, or a full path.")
    parser.add_argument("resolving_power", nargs="?", type=_nonnegative_float, default=0.0,
        help="Resolving power R=lambda/Delta_lambda (default 0: intrinsic).")
    parser.add_argument("vsini", nargs="?", type=_nonnegative_float, default=0.0,
        help="Projected rotational velocity [km/s] (default 0).")
    parser.add_argument("snr", nargs="?", type=_nonnegative_float, default=0.0,
        help="Continuum S/N per displayed pixel (default 0: no noise).")
    parser.add_argument("--seed", type=int, default=None,
        help="Reproduce a specific noise realization; default is new random noise each run.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Tuple[Path, Optional[Path]]:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path.cwd().resolve()
    model_dir = resolve_model_dir(run_dir, Path(args.model))
    inicalc_dir = (run_dir / "../inicalc").resolve()

    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    indat_path = model_dir / "INDAT.DAT"
    formal_path = run_dir / "FORMAL_INPUT"
    if not indat_path.exists():
        raise FileNotFoundError(indat_path)
    if not formal_path.exists():
        raise FileNotFoundError(formal_path)

    params = parse_indat(indat_path)
    _ignored_formal_vsini, formal_lines = parse_formal_input_all(formal_path)

    effective_seed: Optional[int] = None
    if args.snr > 0:
        # Random by default, but print/store the actual 32-bit seed in the PDF
        # so any particular synthetic observation can be reproduced afterwards.
        effective_seed = args.seed if args.seed is not None else secrets.randbits(32)
    elif args.seed is not None:
        warnings.warn("--seed was supplied but S/N=0, so no noise is generated and the seed is ignored")

    line_pdf = make_line_profile_overview(
        run_dir=run_dir, inicalc_dir=inicalc_dir, model_dir=model_dir,
        params=params, resolving_power=args.resolving_power, vsini=args.vsini,
        continuum_snr=args.snr, noise_seed=effective_seed, formal_lines=formal_lines,
    )
    diagnostics_pdf = make_model_diagnostics(run_dir, model_dir, params)

    print(f"Model:              {params.name}")
    print(f"INDAT.DAT used:     {indat_path}")
    print(f"FORMAL_INPUT used:  {formal_path}")
    print(f"Resolving power:    {args.resolving_power if args.resolving_power > 0 else 'intrinsic'}")
    print(f"v sin i:            {args.vsini:g} km/s")
    print(f"Continuum S/N:      {args.snr if args.snr > 0 else 'none'}")
    print(f"Noise seed:         {effective_seed if effective_seed is not None else 'n/a'}")
    return line_pdf, diagnostics_pdf


if __name__ == "__main__":
    main()
