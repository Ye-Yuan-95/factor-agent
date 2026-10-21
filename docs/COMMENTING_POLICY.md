# Commenting and scientific documentation policy

The purpose of comments is to explain the contract and reasons behind the code,
not to repeat each assignment or advertise a favorable research conclusion.

- Every Python module describes its purpose. Statistical modules explain what is
  being measured, what input determines the result and where the method stops.
- Substantive functions document their operation and relevant assumptions,
  missing-data behavior, state mutation or side effects. A name alone is not enough
  for a multistep statistical or orchestration function.
- Inline comments explain non-obvious decisions: label settlement, causal timing,
  seed pairing, denominators, leakage boundaries, solver fallback and missing evidence.
- Data classes and small accessors can rely on descriptive fields/type annotations.
  Test names express assertions; shared fixtures describe what they construct.
- Never state that an experiment confirms a hypothesis before computing results.
  Published reports must contain measurements computed from their identified inputs
  and source, with explicit limits on interpretation.
- Distinguish FPR, realized FDP, expected FDR, full-pipeline power and conditional
  power. A null-panel success is not an alternative-panel FDR guarantee.
- YAML/build metadata use explanatory comments where needed. JSON cannot contain
  comments; its protocols retain descriptive fields and are explained by the guides.
- Changes to comments in source-bound gates still change their byte fingerprints.
  Reissue certificates or use a new run/output directory; do not edit old evidence
  to bypass source checks.

Run `python scripts/audit_comments.py` to check every publishable Python module and
substantive top-level/class function. The twelve-line trigger is a coverage guard,
not a measure of explanation quality. Human review is still needed when behavior
changes. CLI entry points rely on module descriptions, help and the research guide;
nested local closures rely on their enclosing function contract.

Keep generated inventories, validation summaries, research conclusions and
milestones outside this checkout. Public documentation describes commands,
interfaces, input requirements and how to interpret output fields. To save a
private inventory, pass an absolute path outside the repository to
`python scripts/audit_comments.py --report`; see [TESTING.md](TESTING.md).
