# Test suite for Bayesian experiment code

This project includes a pytest test suite for the Bayesian adaptive experiment utilities extracted from the notebook `toy_bayesian_model.ipynb`.

Contents:
- `bayes_experiment.py`: reusable module with functions to test.
- `tests/test_bayes_experiment.py`: unit tests covering core behavior.

Setup (one-time):
1) Create/activate a Python 3.10+ environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Notes:
- Tests use SciPy's Simpson integration for normalization checks and require `numpy` and `scipy`.
- If tests fail due to numerical tolerances on different platforms, consider relaxing tolerances slightly.