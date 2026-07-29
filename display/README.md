# Public display

This directory is the lightweight, reader-facing entry point for the Quantum Turing Patterns movies. The interactive notebooks and display-generation tools are kept separately in [`../interactive-display/`](../interactive-display/), while the paper-scale numerical workflow is in [`../reproduction/`](../reproduction/).

<p align="center">
  <img
    src="assets/labyrinth-animation.gif"
    alt="Formation of a labyrinth quantum Turing pattern"
    width="640"
  >
</p>

<p align="center"><em>Labyrinth pattern formation — continuously looping GIF.</em></p>

## Interactive explorer

- **[Portable Explorer](../interactive-display/qtp_explorer_portable.ipynb)** — standard Python-kernel entry point.
- **[Research Explorer](../interactive-display/qtp_explorer.ipynb)** — full interactive controls.
- **[Exhibit Notebook](../interactive-display/qtp_exhibit.ipynb)** — presentation view of the reference movies.

To launch the full explorer from the repository root:

```bash
cd interactive-display
conda env create -f environment.yml
conda activate qtp-display
python launch_qtp_explorer.py
```

For a standard Jupyter environment:

```bash
cd interactive-display
jupyter lab qtp_explorer_portable.ipynb
```

## Reference movies

- [Labyrinth MP4](../interactive-display/display/display_labyrinth_pattern.mp4)
- [Spot MP4](../interactive-display/display/display_spot_pattern.mp4)
- [Stripe MP4](../interactive-display/display/display_stripe_pattern.mp4)

The GIF above is embedded directly for continuous playback on GitHub. The MP4 files provide the full-resolution outputs.

For numerical regeneration, reference data, and scientific verification, use [`../reproduction/`](../reproduction/).
