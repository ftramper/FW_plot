# FW_plot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22706437.svg)](https://doi.org/10.5281/zenodo.22706437)

A standalone Python command-line utility for plotting **FASTWIND** line profiles and model diagnostics.

The script reads an existing FASTWIND model directory, produces a line-profile overview, and produces a model-diagnostics overview. The line profiles can optionally be processed to mimic an observation through rotational broadening, instrumental broadening, detector sampling, and noise.

This repository contains only the plotting utility and example output; it does **not** contain or distribute FASTWIND itself.

## Features

- Reads stellar, wind, abundance, and clumping parameters from `INDAT.DAT`.
- Discovers available `OUT.*` line-profile files automatically and sorts the panels from blue to red using the wavelength coverage of each profile.
- Reads active and commented line definitions from `FORMAL_INPUT` for component markers.
- Uses FASTWIND atomic data, when available, to mark individual line components.
- Plots model diagnostics from available FASTWIND output files.
- Optional rotational broadening using `PyAstronomy.pyasl.rotBroad`.
- Optional instrumental Gaussian broadening at constant resolving power.
- Samples broadened spectra at three pixels per resolution element.
- Optional photon-dominated noise for a requested continuum S/N.
- Reproducible noise realizations through `--seed`.
- Uses Matplotlib's non-interactive `Agg` backend, making it suitable for batch use.

## Requirements

Recommended: Python 3.9 or newer.

Python packages:

```text
numpy
matplotlib
astropy
PyAstronomy
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

FASTWIND itself and the relevant FASTWIND model/output files must already be available separately.

## FASTWIND directory assumptions

Run the script from the FASTWIND work directory containing `FORMAL_INPUT`.

The `MODEL` argument must point to a model directory containing `INDAT.DAT` and the relevant FASTWIND output files.

For atomic component markers, the script searches for `ATOM_FILE` in:

1. the current FASTWIND work directory, or
2. `../inicalc`

Referenced atomic-data files are then searched for in the work directory, `../inicalc`, and `../inicalc/DATA`.

The binary `MODEL` diagnostic reader currently assumes the FASTWIND v10 sequential-unformatted record layout implemented in the script.

## Usage

```bash
python plot_fw_model.py MODEL [R] [VSINI] [SNR] [--seed SEED]
```

where:

- `MODEL` is the model directory, either relative to the current work directory or an absolute path.
- `R` is the resolving power, `R = lambda / Delta lambda`. Use `0` for no instrumental broadening.
- `VSINI` is the projected rotational velocity in km/s. Use `0` for no rotational broadening.
- `SNR` is the continuum signal-to-noise ratio per displayed observational pixel. Use `0` for no noise.
- `--seed` optionally fixes the random noise realization.

Examples:

```bash
python plot_fw_model.py O9V
python plot_fw_model.py O9V 7500
python plot_fw_model.py O9V 7500 85
python plot_fw_model.py O9V 7500 85 100
python plot_fw_model.py O9V 7500 85 100 --seed 4039807730
```

The synthetic-observation processing is:

```text
intrinsic profile
    -> rotational broadening
    -> instrumental broadening
    -> detector sampling
    -> noise
```

The line-profile calculation starts from FASTWIND's intrinsic `PROF` column rather than `PROFROT`.

## Output

Each run writes a line-profile PDF and, when diagnostic files are available, a diagnostics PDF into the model directory.

Examples of output filenames are:

```text
O9V_line_profiles.pdf
O9V_line_profiles_R7500.pdf
O9V_line_profiles_R7500_vsini85.pdf
O9V_line_profiles_R7500_vsini85_SNR100.pdf
O9V_model_diagnostics.pdf
```

The exact line-profile filename reflects the processing options that were enabled.

## Example output

The [`examples`](examples/) directory contains a sequence for the same `O9V` FASTWIND model, chosen to show the effect of each optional synthetic-observation step separately:

- [`O9V_line_profiles.pdf`](examples/O9V_line_profiles.pdf) — intrinsic FASTWIND profiles.
- [`O9V_line_profiles_R7500.pdf`](examples/O9V_line_profiles_R7500.pdf) — instrumental broadening to `R = 7500`.
- [`O9V_line_profiles_R7500_vsini85.pdf`](examples/O9V_line_profiles_R7500_vsini85.pdf) — `R = 7500` plus `v sin i = 85 km/s` rotational broadening.
- [`O9V_line_profiles_R7500_vsini85_SNR100.pdf`](examples/O9V_line_profiles_R7500_vsini85_SNR100.pdf) — the same processed spectrum with continuum `S/N = 100` per displayed pixel. The stored example uses seed `4039807730`.
- [`O9V_model_diagnostics.pdf`](examples/O9V_model_diagnostics.pdf) — model structure, clumping, flux conservation, convergence, and radiative-acceleration diagnostics for the same model.

The four line-profile examples can be reproduced with:

```bash
python plot_fw_model.py O9V
python plot_fw_model.py O9V 7500
python plot_fw_model.py O9V 7500 85
python plot_fw_model.py O9V 7500 85 100 --seed 4039807730
```

See [`examples/README.md`](examples/README.md) for a compact description of the sequence.

## Notes

Diagnostic panels are created from whichever relevant FASTWIND files are present, including files such as `TAU_ROS`, `TEMP`, `MODEL`, `CLUMPING_OUTPUT`, `FLUXCONT`, `CONVERG`, `CONVERG_METALS`, `MAXTCORR.dat`, and `GRAD.OUT`. Missing diagnostic inputs are handled gracefully where possible.

The script is intended as a practical plotting/inspection tool for FASTWIND model output rather than as part of the FASTWIND distribution itself.

## Citation

If you use **FW_plot** in your work, please cite the software release:

> Tramper, F. (2026). *FW_plot* (v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22706437

GitHub also provides citation metadata through [`CITATION.cff`](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
