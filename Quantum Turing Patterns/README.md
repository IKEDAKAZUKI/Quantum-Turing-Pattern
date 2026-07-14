# Quantum Turing Patterns — Demonstration

This release contains the software and reference results for the three display
cases associated with **Quantum Turing Patterns: A Constructive Quantum Markov
Realization**. It includes an interactive research notebook, a demonstration,
the numerical kernels used to generate the results, and the figures, movies,
summaries, and integrity records needed to verify them.

**Author:** Kazuki Ikeda  
**License:** MIT  
**Release:** 1.0.0

## Package contents

- `qtp_explorer.ipynb` provides the interactive research interface.
- `qtp_explorer_portable.ipynb` runs with a standard Python kernel.
- `qtp_exhibit.ipynb` presents the Spot, Labyrinth, and Stripe movies without
  notebook code or research controls.
- `display/` contains the bundled reference figures, movies, numerical
  summaries, thumbnails, and verification records.
- `qtp_kernels.py`, `qtp_observables.py`, and `qtp_display.py` contain the
  numerical and visualization code used to reproduce the display results.

The scope of this archive is the three display reference cases. Manuscript
source files and analyses unrelated to those cases are not included.

## Installation

The release is tested with Python 3.13. A Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate qtp-display
```

A standard virtual environment can instead use:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Browser checks use Chromium. Install the Playwright-managed browser once when
needed:

```bash
python -m playwright install chromium
```

## Research explorer

Start the interactive notebook with:

```bash
python launch_qtp_explorer.py
```

The explorer varies the initial conditions, follows the time evolution, and can
save a figure, numerical summary, movie, and run manifest. The portable
notebook runs with a standard Python kernel and uses the default preview settings
when interactive controls are unavailable.

Convenience launchers are provided as `start_qtp_explorer.sh` and
`start_qtp_explorer.bat`.

## Exhibit

Start the exhibit with:

```bash
python launch_qtp_exhibit.py
```

The exhibit shows the three precomputed reference movies and exposes only the
pattern selector and playback controls. For an installation, run the complete
browser test on the target computer:

```bash
python verify_browser_interfaces.py --require-live-exhibit
```

Convenience launchers are provided as `start_qtp_exhibit.sh` and
`start_qtp_exhibit.bat`.

## Verify the bundled release

Run these commands from the package directory:

```bash
python verify_release.py --pristine
python verify_display.py --out .
python verify_observable_contract.py
python verify_notebook_package.py --execute-standard --preview-test
python verify_browser_interfaces.py
python check_qtp_environment.py --reference-only
```

The read-only checks write reports to `verification_runtime/`. That directory
is generated locally and is not part of the release manifest.

## Reproduce the reference assets

The bundled files in `display/` are left unchanged by the reproduction command.
Generate a fresh set in `reproduction_output/display/` with:

```bash
python make_display_assets.py
```

This is a full-resolution calculation. The Stripe case is the most computationally
intensive and may take substantially longer than the other two cases.

For the closest match to the release environment, install the base requirements
and then apply the exact core versions used for the bundled results:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-reference.txt
```

The reference Python and library versions are recorded in `BUILD_INFO.json`.
The scientific parameters, random seeds, frame times, diagnostic definitions,
and verification tolerances are also recorded in the generated JSON summaries.

Verify the reproduced files and compare their numerical results with the
bundled references:

```bash
python verify_display.py \
  --out . \
  --display-dir reproduction_output/display \
  --report-dir reproduction_output/verification \
  --reference-dir display
```

Numerical quantities are compared with explicit tolerances. PNG and MP4 bytes
can vary with font, renderer, or codec versions even when the numerical results
agree.

Before building another distribution archive, remove generated output and
caches, regenerate the manifests, and confirm that the tree is clean:

```bash
rm -rf verification_runtime reproduction_output __pycache__
find . -name "*.pyc" -delete
python make_manifest.py .
python verify_release.py --pristine
```

## Citation and license

Citation metadata are provided in `CITATION.cff`. The software is released
under the MIT License; see `LICENSE`.
