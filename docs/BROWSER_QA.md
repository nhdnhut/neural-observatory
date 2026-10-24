# Optional browser regression checks

The application itself requires no Node or packages. This optional test uses Playwright to verify the delivered offline laboratory.

Install Playwright and its Chromium browser in a separate development environment, then run `node scripts/check_browser.cjs` from the project. An existing browser can be supplied using `CHROMIUM_EXECUTABLE`; `NOBS_QA_DIR` controls temporary screenshots. Neither is needed to use the dashboard.

Checks cover Python/JavaScript prediction parity on all 82 included samples; input-saliency finite differences; checkpoints; INT8 selection; neuron clamps and reset; sample filters; drawing; all four tabs; budget recommendations; CSV export; no network requests; and 320/390 px layout. The recorded result is `artifacts/browser_validation.json`. Screenshots are internal inspection outputs in the system temporary directory.

The delivered run used Chrome Headless Shell 155.0.8059.84. The notebooks are guides with unexecuted cells; raw training artifacts and validation logs record the actual completed execution.
