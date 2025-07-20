# AGN_decompose v2.1.0.dev 🚀

2D AGN‑Host decomposition via bidimensional photometric decomposition.

---

## 📖 Description

AGN_decompose is a Python-based tool designed for the two-dimensional decomposition of integral field spectroscopic (IFU) images of galaxies hosting active galactic nuclei (AGN). It uses an iterative method to separate the nuclear AGN emission from the host galaxy, yielding clean spectra and spatial maps of both components ([academic.oup.com](https://academic.oup.com/mnras/article-pdf/536/1/752/61021285/stae2623.pdf?utm_source=chatgpt.com)).

This tool is ideal for analyzing astronomical data and studying AGN properties without contamination from the surrounding galaxy.
If you find usefull this tool pleas cite the paper:

```bibtex
@ARTICLE{Ibarra-Medel2025,
       author = {{Ibarra-Medel}, H. and {Negrete}, C.~A. and {Lacerna}, I. and {Hern{\'a}ndez-Toledo}, H.~M. and {Cortes-Su{\'a}rez}, E. and {S{\'a}nchez}, S.~F.},
        title = "{An iterative method to deblend AGN-Host contributions for Integral Field spectroscopic observations}",
      journal = {\mnras},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = jan,
       volume = {536},
       number = {1},
        pages = {752-776},
          doi = {10.1093/mnras/stae2623},
archivePrefix = {arXiv},
       eprint = {2411.13270},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.536..752I},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```


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

### Documentation: `config_file.yaml` & command line execution

This file contains the configuration settings for executing the **AGN decomposition** (`AGNdecomp`) comand:

```bash
run_agndec run --config_file config_file.yaml
```

Below is a detailed explanation of each variable defined under the `config` section.

---

#### 🔧 Configuration Parameters

| Parameter   | Description |
|-------------|-------------|
| `run_type`  | Type of execution. Set to `FULL` to run the full decomposition pipeline. |
| `name`      | Identifier name for the IFS (Integral Field Spectroscopy) data cube. |
| `path`      | Directory path where the input data cubes are located. |
| `verbose`   | Enables verbose output if set to `True`. Useful for debugging and detailed logging. |
| `ncpus`     | Number of CPU cores to use for parallel processing. |
| `nameb`     | File name pattern for the input data cube. Typically includes the `.cube.fits.gz` extension. |
| `test`      | If `True`, runs the pipeline in test mode (possibly with reduced resolution or steps). |
| `sampling`  | Spectral sampling interval for the analysis in Angstroms (e.g., 1 Å per pixel). |
| `pipeline`  | If `True`, runs the execution in pipeline/batch mode. |
| `clean`     | If `True`, forces a clean run by removing intermediate files or previous results. |
| `psamp`     | Spectral sampling value used in the previous phase. Default is 10 Å. |
| `path_out`  | Output directory where the results and products will be stored. |
| `head_dat`  | Extension index for the primary data in the FITS file (default is 0). |
| `head_err`  | Extension index for the error data in the FITS file (default is 1). |
| `mod_ind`   | Index indicating which model configuration to use for the current run. |
| `mod_ind0`  | Index for the model used during the initial phase of the pipeline. |
| `ext_name`  | Name of the external user-defined function module to be imported and used. |
| `ext_path`  | Path to the directory where the external user module resides. |
| `ext_file`  | Filename of the external Python module (e.g., `extern_file.py`) containing the function. |

---

### 🗂 Notes

- This file is parsed by the `AGNdecomp` command line to set runtime behaviors, I/O paths, sampling configuration, and module extensions.
- Custom user-defined behavior can be introduced via `ext_name`, `ext_path`, and `ext_file`.

---

### ✅ Usage

Place this file in your working directory and modify the parameters to fit your dataset and analysis goals. Then execute the AGN decomposition script that reads this configuration to run the analysis.

---

### Documentation: Writing External Functions for AGN Decomposition

This guide explains how to create and use custom external functions in the `AGNdecomp` pipeline for modeling the light distribution of AGN and host galaxies. It covers the expected function structure, required parameters, and example models including **Moffat** and **Gaussian** profiles.

---

#### 🧩 Purpose of External Functions

External functions allow users to define their own models for point spread functions (PSF) or galaxy light profiles. These models can be used in place of or in addition to built-in profiles, providing flexibility for special cases or new scientific requirements.

---

#### 📁 File Format and Structure

Create a `.py` file with the following requirements:

- Must contain at least one main function that returns a model array.
- Optionally, include a second function to return derived values such as PSF FWHM or total flux.
- Must be importable from the location specified in your `config_file.yaml`.
- It requires the prior definitions in `priors_prop.yaml` to set initial guesses and bounds for the parameters.

#### Required Imports

```python
import numpy as np
import AGNdecomp.tools.tools as tol  # for elliptical radius calculations
```

---

#### 🔧 Example 1: Full Moffat Profile plus Sersic

```python
def ExternFunctionName(pars, x_t=0, y_t=0):
    At=pars['At']
    ds=pars['alpha']
    be=pars['beta']
    dx=pars['xo']
    dy=pars['yo']
    Io=pars['Io']
    bn=pars['bn']
    Re=pars['Re']
    ns=pars['ns']
    es=pars['ellip']
    th=pars['theta']
    r1=tol.radi_ellip(x_t-dx,y_t-dy,es,th)
    model = At * (1.0 + (r1**2.0 / ds**2.0))**(-be)
    model = Io * np.exp(-bn*((r1/Re)**(1./ns) -1.0)) + model
    return model
```

#### Required: PSF total Flux & FWHM Functions

```python
def ExternFunctionName_flux_psf(pars, x_t=0, y_t=0):
    psf = pars['alpha'] * 2.0 * np.sqrt(2.0**(1.0 / pars['beta']) - 1)
    flux = np.pi * pars['alpha']**2.0 * pars['At'] / (pars['beta'] - 1.0)
    return psf, flux
```

---

#### 🔧 Example 2: Gaussian Profile

```python
def GaussianModel(pars, x_t=0, y_t=0):
    A = pars['At']
    sigma = pars['sigma']
    xo = pars['xo']
    yo = pars['yo']
    ellip = pars['ellip']
    theta = pars['theta']

    r = tol.radi_ellip(x_t - xo, y_t - yo, ellip, theta)
    model = A * np.exp(-0.5 * (r / sigma)**2)
    return model

def GaussianModel_flux_psf(pars, x_t=0, y_t=0):
    sigma = pars['sigma']
    A = pars['At']
    fwhm = 2.3548 * sigma
    flux = 2 * np.pi * sigma**2 * A
    return fwhm, flux
```

---

#### 🔄 Integration Steps

1. Define your external model in a `.py` file.
2. Update your `config_file.yaml`:

```yaml
ext_name: GaussianModel
ext_file: extern_function_example.py
ext_path: path/to/your/file/
```

3. Ensure the file is placed in the directory specified in `ext_path`.
4. Define the priors for your model in `priors_prop.yaml`.

---

#### 📌 Parameters

Your external function will receive:

- `pars`: Dictionary containing model parameters as defined in `priors_prop.yaml`.
- `x_t`, `y_t`: 2D arrays (meshgrids) representing spatial pixel coordinates.

Use `tol.radi_ellip(x, y, e, theta)` to compute elliptical distances for modeling elliptical shapes.

---

#### 📚 Tips

- You can test your function independently by calling it with mock `pars` and coordinate grids.
- Use vectorized NumPy operations for performance.
- Return `float32` arrays if memory efficiency is critical.

---

#### 📬 Support

For more advanced use cases or to contribute your functions, refer to the [AGN_decompose GitHub repository](https://github.com/hjibarram/AGN_decompose) or contact the maintainer.

---

### Documentation: `priors_prop.yaml`

This YAML file defines the prior configuration for the model parameters used in a photometric decomposition, the package has a predefined model named `moffat`, and its prior configuration can be found [here](https://github.com/hjibarram/AGN_decompose/blob/main/AGNdecomp/configfiles/priors_prop.yml). Each parameter includes a LaTeX-compatible plot label, an initial guess, and lower/upper bounds for the fitting process with the next structure:

```yaml

--- 
models:
  - name: MODEL_NAME 
    parameters:   
      - name: Par1
        name_plot: '$Par_1$'
        ini_value: InitValue  
        inf_value: LoweBound  
        sup_value: UpperBound  

```

---

### 📌 Notes

- All parameters are provided with an `ini_value` (initial guess), `inf_value` (minimum allowed), and `sup_value` (maximum allowed).
- The `name_plot` field uses LaTeX syntax to allow rendering of mathematical symbols in plots or graphical output.
- This configuration is likely used in an automated fitting pipeline or modeling framework.

---

#### 🔧 Example: Model `moffat`

The `moffat` model is commonly used to describe point spread functions (PSFs) in astronomical imaging, this model is predefined within the package, but the user can define his own models with the use of `ext_file`. The parameters listed below are likely used for 2D image fitting or profile modeling.

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
