# AGN_decompose v2.1.0.dev 🚀

2D AGN‑Host decomposition via bidimensional photometric decomposition.

---

## 📖 Description

AGN_decompose is a Python-based tool designed for the two-dimensional decomposition of integral field spectroscopic (IFU) images of galaxies hosting active galactic nuclei (AGN). It uses an iterative method to separate the nuclear AGN emission from the host galaxy, yielding clean spectra and spatial maps of both components ([academic.oup.com](https://academic.oup.com/mnras/article-pdf/536/1/752/61021285/stae2623.pdf?utm_source=chatgpt.com)).

This tool is ideal for analyzing astronomical data and studying AGN properties without contamination from the surrounding galaxy.

---

## 📦 Repository Structure

```
.
├── AGNdecomp/           # Core module with classes and functions
├── Examples/            # Example usage scripts
├── bin/                 # Command-line utilities
├── setup.py             # Installer script
├── pyproject.toml       # Dependency management
└── README.md            # This file
```

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/hjibarram/AGN_decompose.git
cd AGN_decompose
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the package:

```bash
pip install .
```

Or:

```bash
pip install -e .
```

---

## 🎯 Main Features

- **2D decomposition**: Iteratively separates AGN and host components.
- **IFU-ready**: Designed for integral field spectroscopy data.
- **Modular workflow**: Organized into reusable functions under `AGNdecomp/`.
- **Usage examples**: Step-by-step guides under `Examples/`.

---

## 🚀 Basic Usage

After installation, you can import the module and perform decomposition:

```python
from AGNdecomp import Decomposer

# Initialize with IFU image or datacube
decomp = Decomposer('path/to/image.fits')

# Run decomposition
results = decomp.run()

# Save separated spectra
results.save_spectra('AGN.fits', 'host.fits')

# Plot component maps
results.plot_maps('AGN_map.png', 'host_map.png')
```

See the `Examples/` directory for real use cases (e.g., `decompose_example.py`).

---

## 📂 Directory Overview

- `AGNdecomp/`: Core logic and decomposition engine.
- `Examples/`: Practical scripts demonstrating functionality.
- `bin/`: Command-line support tools.
- `setup.py` and `pyproject.toml`: Package management.

---

## 📖 Documentation & Resources

- Inline docstrings and comments inside `__init__.py` and other modules.
- Examples provide a step-by-step entry point.
- For theoretical background, refer to the associated paper:
  [https://academic.oup.com/mnras/article/536/1/752/7907265](https://academic.oup.com/mnras/article/536/1/752/7907265)

---

## 🧪 Testing & CI

- Unit tests are not included yet—recommended for future development.
- No CI configured (e.g., GitHub Actions), but encouraged for scalability.

---

## 🗣 Contributing

Contributions are welcome!

- Report issues or bugs.
- Suggest improvements or features.
- Submit pull requests (with documentation and test coverage when possible).

---

## 📄 License

(Insert license here if available or clarify open/restricted usage.)

---

## 📬 Contact

For questions, comments, or collaboration:
**hjibarram** via [GitHub profile](https://github.com/hjibarram).

---

## 🔭 Roadmap

- Support for other formats (e.g., JWST/MUSE datacubes).
- Custom PSF fitting options.
- Interactive map visualizations.

---

*This README serves as a complete guide and will evolve with future updates.*
