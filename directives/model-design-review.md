# Model Design Review

Tags: #directive #model-design #archimedes

Use when model architecture, data behavior, report generation, QA results, or evaluation strategy comes up.

## Steps

1. Identify the decision, experiment, or failure mode being discussed.
2. Connect it back to [[second-brain/projects/model-design]].
3. Check whether the issue affects data correctness, user trust, explainability, or reproducibility.
4. Preserve durable lessons in daily memory or the project hub.
5. Recommend the smallest test or implementation change that would reduce risk.

## Watch For

- Silent data loss
- Bad joins or grain mismatches
- Metrics that look plausible but are wrong
- Report language that overclaims
- Evaluation gaps
- Changes that improve demos but weaken real customer behavior

