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
from AGNdecomp.tools.prof_fit import prof_an

# Initialize with IFU image or datacube
cube_data, hdr0=fits.getdata('path/to/image.fits', 'FLUX', header=True)
cube_dataE=fits.getdata('path/to/image.fits', 'ERROR', header=False)


# Run decomposition
prof_ana(cube_data,cube_dataE,hdr0,name='NAME',ncpu=ncpus)

```

See the `Examples/` directory for real use cases (e.g., `Example.ipynb`).

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
# Documentation: `priors_prop.yaml`

This YAML file defines the prior configuration for the model parameters used in a photometric decomposition, the package has a predefined model named `moffat`. Each parameter includes a LaTeX-compatible plot label, an initial guess, and lower/upper bounds for the fitting process.

---

## 🔧 Model: `moffat`

The `moffat` model is commonly used to describe point spread functions (PSFs) in astronomical imaging. The parameters listed below are likely used for 2D image fitting or profile modeling.

| Parameter | Plot Label | Description | Initial Value | Lower Bound | Upper Bound |
|-----------|------------|-------------|----------------|-------------|-------------|
| `At`      | `$A_t$`     | Amplitude or normalization factor of the Moffat profile. | 0.22 | 0.1 | 0.5 |
| `alpha`   | `$\alpha$` | Moffat parameter controlling the width of the core. | 2 | 0 | 10 |
| `beta`    | `$\beta$`  | Moffat parameter that controls the wings of the profile. | 2 | 0 | 10 |
| `xo`      | `$X_0$`     | X-coordinate of the profile center. | 0 | -10 | 10 |
| `yo`      | `$Y_0$`     | Y-coordinate of the profile center. | 0 | -10 | 10 |
| `Io`      | `$I_e$`     | Effective intensity, typically at the effective radius. | 0.01 | 0 | 10 |
| `bn`      | `$b_n$`     | Sersic parameter related to profile concentration. | 2 | 0.5 | 20 |
| `Re`      | `$R_e$`     | Effective radius (radius enclosing half of the light). | 4 | 0 | 30 |
| `ns`      | `$n_s$`     | Sersic index defining the shape of the light profile. | 2 | 0 | 15 |
| `ellip`   | `$e_s$`     | Ellipticity of the model (0 = circular). | 0 | 0 | 10 |
| `theta`   | `$\theta_s$` | Position angle of the model in degrees. | 0 | 0 | 180 |

---

## 📌 Notes

- All parameters are provided with an `ini_value` (initial guess), `inf_value` (minimum allowed), and `sup_value` (maximum allowed).
- The `name_plot` field uses LaTeX syntax to allow rendering of mathematical symbols in plots or graphical output.
- This configuration is likely used in an automated fitting pipeline or modeling framework.

---

## ✅ Usage

This file can be used as a configuration input for scripts that perform 2D fitting, such as galaxy decomposition or PSF modeling, by parsing these parameter priors.

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

- Custom PSF fitting options.
- Interactive map visualizations.

---

*This README serves as a complete guide and will evolve with future updates.*
