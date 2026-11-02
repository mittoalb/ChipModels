# ChipModels

Synthetic 3-D phantoms of silicon microchip packages for tomographic
reconstruction, image-segmentation benchmarks, and physics-based rendering
experiments.

## Scope

* `chip.py` — single-script chip-package generator: layered FEOL / BEOL stack,
  copper interconnect metalisation, bump arrays, through-silicon vias, an
  encapsulation lid; each labelled with its own voxel class for downstream
  simulation.
* `chip_codes.json` — label → (alias, elemental composition, bulk density)
  mapping, same schema as the brain phantom.

## Install

```bash
pip install -e .
```

## Usage

Edit the parameters at the top of `chip.py` (device size, metal layer count,
bump pitch, etc.) and run:

```bash
python chip.py
```

Produces an OME-Zarr volume under `data/chip_model.zarr`. Open it with any
Zarr viewer:

```bash
# vedo-based volume viewer, bundled with MSim
python /path/to/MSim/msim/vis_volume.py data/chip_model.zarr
```

## Status

Research code — API is in flux. Pull requests welcome.

## License

Same as the parent TomoModels project. See `LICENSE` once added.

## Attribution

Original author: Alberto Mittone (@mittoalb). History preserved via
`git-filter-repo` from TomoModels' `Semiconductors/` subtree.
