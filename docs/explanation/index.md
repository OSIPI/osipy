# Explanation

## Software Architecture

- [Architecture Overview](architecture.md) — Module structure, data flow, and registry-driven extensibility.
- [Registry-Driven Configuration](configuration.md) — How the CLI/YAML config is generated from the registries via `MethodConfig` schemas.
- [The xp Abstraction Pattern](xp-abstraction.md) — GPU/CPU agnostic code using `xp = get_array_module()`.
- [OSIPI Standards](osipi-standards.md) — CAPLEX naming, units, and validation against DROs.

## MRI Physics & Theory

- [Pharmacokinetic Models](pharmacokinetic-models.md) — Tofts, Extended Tofts, Patlak, and Two-Compartment models for DCE-MRI.
- [ASL Physics](asl-physics.md) — Arterial spin labeling, from magnetic labeling to CBF quantification.
- [IVIM Theory](ivim-theory.md) — The bi-exponential signal model separating diffusion and perfusion.
- [DSC Deconvolution](dsc-deconvolution.md) — SVD-based deconvolution and how perfusion parameters are derived.

## Reading Guide

| If you want to understand... | Read |
|------------------------------|------|
| How the code is organized | [Architecture Overview](architecture.md) |
| Why the YAML config is shaped the way it is | [Registry-Driven Configuration](configuration.md) |
| How GPU acceleration works | [xp Abstraction Pattern](xp-abstraction.md) |
| What OSIPI standards mean | [OSIPI Standards](osipi-standards.md) |
| How DCE models work mathematically | [Pharmacokinetic Models](pharmacokinetic-models.md) |
| Why ASL doesn't need contrast | [ASL Physics](asl-physics.md) |
| What IVIM measures | [IVIM Theory](ivim-theory.md) |
| How DSC gets perfusion from signal | [DSC Deconvolution](dsc-deconvolution.md) |
