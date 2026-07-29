# Interactive Pattern Explorer

This directory is the reader-facing entry point for exploring the Quantum Turing Patterns display. The Jupyter notebooks run the display code directly, while the numerical workflows used for the paper remain separate in [`../reproduction/`](../reproduction/).

<p align="center">
  <a href="../Quantum%20turing%20patterns-1.0.0/display/display_labyrinth_pattern.mp4">
    <img
      src="https://raw.githubusercontent.com/IKEDAKAZUKI/Quantum-Turing-Pattern/main/display/assets/labyrinth-loop.gif"
      alt="Formation of a labyrinth quantum Turing pattern"
      width="640"
    >
  </a>
</p>

<p align="center"><em>Labyrinth pattern formation — continuously looping GIF preview. Click the animation to open the full-resolution MP4.</em></p>

## Run the Jupyter explorers

### Portable Explorer — recommended

[`qtp_explorer_portable.ipynb`](../Quantum%20turing%20patterns-1.0.0/qtp_explorer_portable.ipynb) uses a standard Python kernel. It lets readers explore the three reference patterns, follow their time evolution, and save selected results. When interactive widgets are unavailable, it runs a preview with the default settings in an ordinary notebook cell.

```bash
cd "Quantum turing patterns-1.0.0"
conda env create -f environment.yml
conda activate qtp-display
jupyter lab qtp_explorer_portable.ipynb
```

### Research Explorer — full controls

[`qtp_explorer.ipynb`](../Quantum%20turing%20patterns-1.0.0/qtp_explorer.ipynb) provides the full widget-based interface for varying initial conditions, inspecting the evolution, and saving figures, movies, and numerical summaries. The launcher verifies the interactive environment and installs the dedicated `QTP Display` kernel when needed.

```bash
cd "Quantum turing patterns-1.0.0"
conda env create -f environment.yml
conda activate qtp-display
python launch_qtp_explorer.py
```

### Exhibit Notebook — presentation mode

[`qtp_exhibit.ipynb`](../Quantum%20turing%20patterns-1.0.0/qtp_exhibit.ipynb) presents the precomputed stripe, spot, and labyrinth movies without the research controls.

## Reference movies

- [Labyrinth MP4](../Quantum%20turing%20patterns-1.0.0/display/display_labyrinth_pattern.mp4)
- [Spot MP4](../Quantum%20turing%20patterns-1.0.0/display/display_spot_pattern.mp4)
- [Stripe MP4](../Quantum%20turing%20patterns-1.0.0/display/display_stripe_pattern.mp4)

The GIF above is optimized for continuous playback on GitHub. The MP4 files retain the full-resolution display output.

For paper-scale regeneration, reference data, and scientific verification, use [`../reproduction/`](../reproduction/).
