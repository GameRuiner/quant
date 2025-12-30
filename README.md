# Quant Project

## Project Structure

```
quant/
├── src/
│   ├── ar.py
│   └── ma.py
├── tests/
│   └── ar_test.py
└── README.md
```

## How to Run the Tests

1. **Install dependencies**  
   Make sure you have Python 3 and the required packages:
   ```
   pip install numpy pytest statsmodels
   ```

2. **Run the tests using pytest**  
   From the root of the project, run:
   ```
   pytest tests/
   ```

   To see log output, run:
   ```
   pytest -s --log-cli-level=INFO tests/
   ```

3. **What the tests do**  
   The tests compare the coefficients estimated by your custom AR(1) implementation (`src/ar.py`) with those from `statsmodels`'s `AutoReg` on synthetic data.

## Notes

- Make sure your `src` directory contains your model implementations.
- You can add more tests in the `tests/` directory as your project grows.