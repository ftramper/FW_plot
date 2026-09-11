# Example output

These PDFs show the same FASTWIND `O9V` model at successive stages of the optional synthetic-observation processing. This makes it easy to see separately what instrumental resolution, rotational broadening, and noise do to the intrinsic line profiles.

## Line-profile sequence

- [`O9V_line_profiles.pdf`](O9V_line_profiles.pdf)  
  Intrinsic FASTWIND line profiles (`R = intrinsic`, `v sin i = 0 km/s`, no added noise).

- [`O9V_line_profiles_R7500.pdf`](O9V_line_profiles_R7500.pdf)  
  Instrumentally broadened to `R = 7500`, without rotational broadening or noise.

- [`O9V_line_profiles_R7500_vsini85.pdf`](O9V_line_profiles_R7500_vsini85.pdf)  
  `R = 7500` plus rotational broadening with `v sin i = 85 km/s`, without added noise.

- [`O9V_line_profiles_R7500_vsini85_SNR100.pdf`](O9V_line_profiles_R7500_vsini85_SNR100.pdf)  
  `R = 7500`, `v sin i = 85 km/s`, and continuum `S/N = 100` per displayed observational pixel. The stored example uses noise seed `4039807730`.

The corresponding commands are:

```bash
python plot_fw_model.py O9V
python plot_fw_model.py O9V 7500
python plot_fw_model.py O9V 7500 85
python plot_fw_model.py O9V 7500 85 100 --seed 4039807730
```

## Model diagnostics

- [`O9V_model_diagnostics.pdf`](O9V_model_diagnostics.pdf)  
  Model-structure and convergence diagnostics for the same FASTWIND model.

Together, the files illustrate both the direct FASTWIND output and the script's optional transformation from intrinsic profiles to a simple synthetic observation.
