# Product PRD - Action Reports Workflow - 2026-07-11

## Problem

Users have messy business exports and need a useful report before they know whether the data deserves a full dashboard, integration, or analyst project.

Current alternatives:

- Manual spreadsheet cleanup.
- Generic CSV chat.
- Overbuilt BI setup.
- One-off analyst work.

## Target User

An ecommerce or SaaS operator who owns a recurring business question but does not have clean analysis infrastructure for every export.

## Core Job

Turn a messy operational dataset into a focused action report that answers:

- What happened?
- Where is risk or opportunity concentrated?
- Which rows, accounts, SKUs, or segments should be reviewed first?
- What operating rule should the team test next?
- What data quality issues might make the answer unreliable?

## Must-Have Workflow

1. Upload CSV.
2. Confirm data type and sensitive-data warning.
3. Choose business question or target column.
4. Identify columns to exclude.
5. Generate first-pass report.
6. Show action queue logic.
7. Export/share report.

## Report Sections

- Executive summary.
- Dataset overview.
- Target/business question.
- Key metrics.
- Segment findings.
- Risk/opportunity drivers.
- Recommended action queue.
- Operating rules.
- Data quality and leakage warnings.
- Suggested next analysis.

## Differentiators

- Guided business question selection.
- Leakage/proxy-column warning.
- Action queue output.
- Operating rule output.
- Explicit data-quality caveats.
- Shareable report page.

## Guardrails

- Warn users before regulated or sensitive data upload.
- Encourage anonymized samples.
- Flag likely proxy/leakage columns.
- Avoid unsupported causal claims.
- Avoid medical, legal, or financial advice language.
- Keep human review in the loop.

## MVP Success Metrics

- User completes upload-to-report workflow.
- User shares report.
- User says the report identified a useful review queue or operating rule.
- User submits a second file or asks for a follow-up.

## Open Product Questions

- Should the first commercial version be self-serve software, assisted service, or hybrid?
- Should ecommerce stockout/margin get a specialized report template?
- Should SaaS churn/support get a specialized report template?
- What export formats matter beyond CSV?
- How should reports be shared publicly without exposing data?

