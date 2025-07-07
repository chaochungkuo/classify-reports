# ClassifyAnything: Notebook-based ML Classification Pipeline

A modular, notebook-driven machine learning pipeline for classification tasks.  
Configuration is managed via `config.toml`, and each step is a separate, reproducible notebook.

## Features

- **Flexible Data Input:** Supports CSV/TSV/Excel, with auto-detection of features and labels.
- **Configurable Pipeline:** All steps (imputation, normalization, feature selection, modeling, etc.) are controlled via `config.toml`.
- **Reproducible Environment:** Uses [pixi](https://prefix.dev/docs/pixi/) for dependency management.
- **Automated Notebook Execution:** Run all steps with a single script.
- **Debuggable:** Open any notebook to inspect or rerun steps interactively.

---

## Quickstart

### 0. Clone this repository

```bash
git clone https://github.com/chaochungkuo/classify-reports.git
cd classify-reports
```

### 1. Initiate the pixi environment

```bash
pixi shell
```

This will create and activate a reproducible environment with all dependencies.

### 2. Configure your pipeline

Edit `config.toml` to specify:

- Input data path (`[data].input_path`)
- Sample ID and label columns
- Feature selection and modeling options
- Preprocessing steps (imputation, normalization, log transform, etc.)

See the comments in `config.toml` for detailed options.

### 3. Run the pipeline

```bash
bash render.sh
```

This will execute all notebooks in order, saving outputs and reports to the specified output directory (see `[reporting].output_dir` in `config.toml`).

---

## Debugging

If you encounter errors or want to inspect results:

- Open the relevant notebook (e.g., `notebooks/03_preprocessing.ipynb`) in Jupyter or VSCode.
- Run cells interactively to debug or modify steps.
- Check the output and logs for details.

---

## Output

- Results, reports, and model artifacts are saved to the directory specified in `config.toml` (`[reporting].output_dir`).
- HTML reports and plots are generated if enabled in the config.

---

## Requirements

- [pixi](https://prefix.dev/docs/pixi/) (for environment management)
- Bash (for `render.sh`)

---

## License

[MIT License](LICENSE) (or your preferred license)

---

**For questions or contributions, please open an issue or pull request.**

---

**Important:**
- Do **not** push any analysis results, outputs, or generated files (such as data, reports, or model artifacts) back to GitHub. Keep the repository clean and version-controlled for code and configuration only.
