This directory includes a few sample datasets to get you started.

*   `california_housing_data*.csv` is California housing data from the 1990 US
    Census; more information is available at:
    https://docs.google.com/document/d/e/2PACX-1vRhYtsvc5eOR2FWNCwaBiKL6suIOrxJig8LcSBbmCbyYsayia_DvPOOBlXZ4CAlQ5nlDD8kTaIDRwrN/pub

# Loan Approval Prediction System

A simple repo
# Loan Approval Prediction System

This repository contains a Streamlit app and a trained XGBoost model to predict loan approval decisions based on applicant features.

## Project Overview

- **Purpose:** Demonstrate a simple ML pipeline for predicting loan approvals using tabular applicant data.
- **App:** `streamlit_app.py` — a Streamlit frontend to collect input and display model predictions.
- **Model:** `xgb_model.joblib` — trained XGBoost model.
- **Preprocessing:** `preprocessor.joblib`, `le_target.joblib`, `important_feature_names.joblib` — artifacts used to transform inputs.
- **Data sample:** `loan_approval_data.csv` — example dataset used for training and exploration.

## Quick Start

1. Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS / Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser and let you input applicant details to get a loan approval prediction.

## Files

- `streamlit_app.py` — Streamlit web UI.
- `xgb_model.joblib` — serialized model.
- `preprocessor.joblib` — preprocessing pipeline.
- `le_target.joblib` — label encoder for the target.
- `important_feature_names.joblib` — important features list.
- `loan_approval_data.csv` — sample dataset.

## Notes & Troubleshooting

- If you get errors loading `.joblib` files, confirm the working directory is the repository root.
- If model loading fails due to package-version mismatches, try installing the package versions used during development (see `requirements.txt`).

## Contributing

Contributions and improvements are welcome. Open an issue or a pull request with suggestions or fixes.

## License

This project is provided as-is. Add a license if you intend to reuse or distribute it.

This directory includes a few sample datasets to get you started.

*   `california_housing_data*.csv` is California housing data from the 1990 US
    Census; more information is available at:
    https://docs.google.com/document/d/e/2PACX-1vRhYtsvc5eOR2FWNCwaBiKL6suIOrxJig8LcSBbmCbyYsayia_DvPOOBlXZ4CAlQ5nlDD8kTaIDRwrN/pub

*   `mnist_*.csv` is a small sample of the
    [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), which is
    described at: http://yann.lecun.com/exdb/mnist/

*   `anscombe.json` contains a copy of
    [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet); it
    was originally described in

    Anscombe, F. J. (1973). 'Graphs in Statistical Analysis'. American
    Statistician. 27 (1): 17-21. JSTOR 2682899.

    and our copy was prepared by the
    [vega_datasets library](https://github.com/altair-viz/vega_datasets/blob/4f67bdaad10f45e3549984e17e1b3088c731503d/vega_datasets/_data/anscombe.json).
