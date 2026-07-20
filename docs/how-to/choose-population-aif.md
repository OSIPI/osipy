# How to Choose a Population AIF

Select the appropriate population-based arterial input function for DCE-MRI analysis.

## Via CLI (YAML Config)

Set the `population_aif` block in your pipeline config. The AIF is selected by
its `name`:

```yaml
modality: dce
pipeline:
  model:
    method: extended_tofts
  aif_source: population
  population_aif:
    name: parker  # or georgiou, fritz_hansen, weinmann, mcgrath
```

Available names: `parker`, `georgiou`, `fritz_hansen`, `weinmann`, `mcgrath`. See the characteristics table below to choose.

## Via Python API

osipy provides five population AIF models:

```python
import osipy
import numpy as np

time = np.linspace(0, 300, 100)  # seconds

# Parker AIF (most widely used)
parker = osipy.ParkerAIF()(time)

# Georgiou AIF (broader peak)
georgiou = osipy.GeorgiouAIF()(time)

# Fritz-Hansen AIF (earlier model)
fritz_hansen = osipy.FritzHansenAIF()(time)

# Weinmann AIF (classic bi-exponential)
weinmann = osipy.get_population_aif("weinmann")(time)

# McGrath AIF (preclinical, small animal)
mcgrath = osipy.get_population_aif("mcgrath")(time)
```

## Compare AIF Shapes

Visualize the differences between population AIFs:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time, parker.concentration, 'r-', linewidth=2, label='Parker')
ax.plot(time, georgiou.concentration, 'b-', linewidth=2, label='Georgiou')
ax.plot(time, fritz_hansen.concentration, 'g-', linewidth=2, label='Fritz-Hansen')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (mM)')
ax.set_title('Population AIF Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## AIF Characteristics

| Model | Shape | Best For |
|-------|-------|----------|
| Parker | Dual Gaussian + exponential tail | Standard DCE, high temporal resolution |
| Georgiou | Dual Gaussian + exponential decay | Lower temporal resolution, smoother |
| Fritz-Hansen | Bi-exponential decay | Cardiac perfusion, early studies |
| Weinmann | Bi-exponential decay | Classic reference, Gd-DTPA pharmacokinetics |
| McGrath | Gamma-variate + exponential washout | Preclinical (small animal) studies |

## Selection Guidelines

All five AIFs at a glance:

```python
# Default for most applications
aif = osipy.ParkerAIF()(time)

# For slower acquisitions (TR > 5s)
aif = osipy.GeorgiouAIF()(time)

# For cardiac perfusion or historical comparison
aif = osipy.FritzHansenAIF()(time)

# Classic Gd-DTPA reference
aif = osipy.get_population_aif("weinmann")(time)

# Preclinical (small animal) studies
aif = osipy.get_population_aif("mcgrath")(time)
```

### Use Parker AIF When:

- You have standard DCE acquisition (TR < 5s)
- Measuring tumors, muscle, or general tissues
- This is your first analysis (most validated)

### Use Georgiou AIF When:

- Lower temporal resolution (TR > 5s)
- Want smoother concentration curves
- Bolus injection was slow

### Use Fritz-Hansen AIF When:

- Cardiac perfusion studies
- Comparing with older literature

### Use Weinmann AIF When:

- Reproducing classic Gd-DTPA pharmacokinetic studies
- Comparing with Weinmann et al. (1984) reference data

### Use McGrath AIF When:

- Preclinical (small animal) DCE-MRI studies
- Rodent models where human population AIFs are inappropriate
- Need a gamma-variate + exponential AIF model

## Scale AIF for Injection Dose

Adjust AIF for different contrast agent doses:

```python
from osipy.common.types import AIFType

# Standard dose is 0.1 mmol/kg
# For half dose (0.05 mmol/kg):
dose_factor = 0.05 / 0.1

aif = osipy.ParkerAIF()(time)
aif_scaled = osipy.ArterialInputFunction(
    time=time,
    concentration=aif.concentration * dose_factor,
    aif_type=AIFType.MEASURED,
)
```

## Adjust for Hematocrit

Account for blood hematocrit:

```python
from osipy.common.types import AIFType

def adjust_aif_hematocrit(aif, hct=0.45):
    """Adjust AIF for hematocrit.

    Contrast agent distributes in plasma, not whole blood.
    Standard AIF assumes Hct = 0.45.
    """
    # Plasma fraction
    plasma_fraction = 1 - hct
    standard_plasma = 1 - 0.45
    # Scale concentration
    scale = standard_plasma / plasma_fraction
    adjusted_conc = aif.concentration * scale
    return osipy.ArterialInputFunction(
        time=aif.time,
        concentration=adjusted_conc,
        aif_type=AIFType.MEASURED,
    )

# For patient with Hct = 0.35
aif_adjusted = adjust_aif_hematocrit(parker, hct=0.35)
```

## Compare Fitting Results

Evaluate which AIF works best for your data:

```python
import osipy

# Try different AIFs
aifs = {
    'Parker': osipy.ParkerAIF()(time),
    'Georgiou': osipy.GeorgiouAIF()(time),
}

results = {}
for name, aif in aifs.items():
    result = osipy.fit_model(
        "extended_tofts",
        concentration=concentration,
        aif=aif,
        time=time,
        mask=mask
    )
    results[name] = result

# Compare R² values
for name, result in results.items():
    valid = result.quality_mask > 0
    mean_r2 = result.r_squared_map[valid].mean()
    print(f"{name} AIF: Mean R² = {mean_r2:.4f}")
```

## AIF Properties

Access AIF characteristics:

```python
aif = osipy.ParkerAIF()(time)

# Peak information
print(f"Peak concentration: {aif.peak_concentration:.2f} mM")
print(f"Peak time: {aif.peak_time:.1f} s")

# Access raw data
print(f"Time points: {len(aif.time)}")
print(f"Concentration array: {aif.concentration.shape}")
```

## When to Use Population vs Measured AIF

| Scenario | Recommendation |
|----------|----------------|
| No visible artery in FOV | Population AIF |
| Large artery available | Measured AIF |
| Comparing across subjects | Population AIF (consistent) |
| Absolute quantification | Measured AIF (more accurate) |
| Rapid screening | Population AIF |
| Research publication | Consider both, report which |

## Common Issues

### AIF Peak Too Early/Late

Shift AIF using the shift_aif utility:

```python
from osipy.common.aif import shift_aif

# Shift AIF by a known delay (in seconds)
aif = osipy.ParkerAIF()(time)
shifted_conc = shift_aif(aif.concentration, time, delay=10.0, xp=np)
```

Alternatively, use `fit_delay=True` to estimate the delay automatically during model fitting:

```python
result = osipy.fit_model(
    "extended_tofts", concentration, aif, time,
    fit_delay=True  # Estimates per-voxel delay
)
delay_map = result.parameter_maps["delay"].values  # seconds
```

### Mismatch with Measured Data

Scale AIF to match observed peak concentration:

```python
# If population AIF doesn't match your observed curves,
# consider:
# 1. Different injection rate
# 2. Cardiac output differences
# 3. Acquisition timing issues

# Scale to match observed peak
observed_peak = 4.0  # mM
scale = observed_peak / aif.peak_concentration
aif_scaled = osipy.ArterialInputFunction(
    time=aif.time,
    concentration=aif.concentration * scale,
    aif_type=AIFType.MEASURED,
)
```

## See Also

- [How to Use a Custom AIF](use-custom-aif.md)
- [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md)
- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
