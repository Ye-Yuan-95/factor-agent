# Testing and package checks

Run checkout commands below from the repository root. Test fixtures generate
their own inputs; market datasets and LLM credentials are unnecessary.

## Dummy workflow

Install core and test dependencies, then execute the README workflow test:

```sh
python -m pip install -e ".[dev]"
python -m pytest tests/test_cli.py::test_readme_dummy_workflow -q
```

The test reads the Quick start commands from [README.md](../README.md) and runs
them in a temporary source copy. It checks the calibration panel, separate
synthetic market, configured trial budget, FactorCards, ledger, factor list,
user report, unopened test set and overwrite refusal. It requires no particular
acceptance count or numerical gate estimate. The tutorial's five-trial budget
does not reach the sixth candidate, which deliberately uses a forbidden label;
DSL rejection has separate tests.

To try the workflow interactively, follow the README commands. Use `demo-data
--out <new-path>` and `run --panel-csv <new-path>` to preserve an existing panel.
Generated reports belong in ignored runtime directories or an external research
folder, rather than in public documentation.

## Full test suite

Optimization tests need the optional solver packages:

```sh
python -m pip install -e ".[dev,opt]"
python -m pytest tests -q
```

For experiment entry points only, run
`python -m pytest tests/test_research_experiments.py -q`.
Use small fixtures to check contracts; choose calibration inputs, protocol seeds
and simulation budgets separately for substantive experiments.

## Source documentation

Check Python module descriptions and substantive function docstrings:

```sh
python scripts/audit_comments.py
```

The command prints its inventory summary without creating a report. To save a
private Markdown inventory, pass `--report` with an absolute path outside this
checkout. In-repository report destinations are refused. This is a structural
check; review scientific explanations against the implementation and
[COMMENTING_POLICY.md](COMMENTING_POLICY.md).

## Wheel installation

Build the wheel after installing the dependencies above:

```sh
python -m pip install build
python -m build --wheel
```

Install the resulting wheel with `python -m pip install --force-reinstall
--no-deps <absolute-path-to-wheel>`. Change to a directory outside the checkout
and run `python -I <absolute-path-to-checkout>/scripts/wheel_smoke.py`.
The check verifies installed import locations, package data, console registration,
provenance JSON and module help commands. For an isolated `pip --target`
installation, pass its directory through `--installed-root`.

## Continuous integration

The workflow in `.github/workflows/ci.yml` configures Ubuntu and Windows tests
on Python 3.10 and 3.13. A separate Python 3.13 job runs the README dummy test with
core and test dependencies only. Inspect the current commit's GitHub Actions
checks for execution status. Keep downloaded logs, environment inventories,
validation summaries, conclusions and milestones outside the checkout.
