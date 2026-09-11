# FW_plot

A standalone Python command-line utility for plotting **FASTWIND** line profiles and model diagnostics.

The script reads an existing FASTWIND model directory, produces a line-profile overview, and produces a model-diagnostics overview. The line profiles can optionally be processed to mimic an observation through rotational broadening, instrumental broadening, detector sampling, and noise.

This repository contains only the plotting utility and example output; it does **not** contain or distribute FASTWIND itself.

## Features

- Reads stellar, wind, abundance, and clumping parameters from `INDAT.DAT`.
- Discovers available `OUT.*` line-profile files automatically.
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
python plot_fw_model.py T40g37R15N60
python plot_fw_model.py T40g37R15N60 7500
python plot_fw_model.py T40g37R15N60 7500 100
python plot_fw_model.py T40g37R15N60 7500 100 150
python plot_fw_model.py T40g37R15N60 7500 100 150 --seed 12345
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
T40g37R15N60_line_profiles.pdf
T40g37R15N60_line_profiles_R7500_vsini100.pdf
T40g37R15N60_line_profiles_R7500_vsini100_SNR150.pdf
T40g37R15N60_model_diagnostics.pdf
```

The exact line-profile filename reflects the processing options that were enabled.

## Example output

The repository includes two example PDFs for the model `T40g37R15N60`:

- [`T40g37R15N60_line_profiles_R7500_vsini100.pdf`](examples/T40g37R15N60_line_profiles_R7500_vsini100.pdf) — line-profile overview for `R = 7500`, `v sin i = 100 km/s`, without added noise.
- [`T40g37R15N60_model_diagnostics.pdf`](examples/T40g37R15N60_model_diagnostics.pdf) — model structure and convergence diagnostics.

The line-profile example corresponds to:

```bash
python plot_fw_model.py T40g37R15N60 7500 100
```

## Notes

Diagnostic panels are created from whichever relevant FASTWIND files are present, including files such as `TAU_ROS`, `TEMP`, `MODEL`, `CLUMPING_OUTPUT`, `FLUXCONT`, `CONVERG`, `CONVERG_METALS`, `MAXTCORR.dat`, and `GRAD.OUT`. Missing diagnostic inputs are handled gracefully where possible.

The script is intended as a practical plotting/inspection tool for FASTWIND model output rather than as part of the FASTWIND distribution itself.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
