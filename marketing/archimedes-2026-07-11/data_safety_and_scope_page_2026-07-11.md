# Archimedes Data Safety and Scope Page - 2026-07-11

## What To Send First

For the first review, send synthetic, anonymized, or sample data whenever possible.

Good first files:

- Ecommerce SKU, order, inventory, promotion, or margin exports.
- SaaS account, support, onboarding, churn, or customer-health exports.
- Reporting prep files.
- Demo datasets.
- CSV files where the business question is clear enough to describe in one paragraph.

## What Not To Send

Do not send:

- Medical records.
- Regulated healthcare data.
- Full credit decision files.
- Social Security numbers.
- Payment card data.
- Passwords, tokens, API keys, or secrets.
- Private customer messages unless anonymized.
- Any data you do not have permission to share.

## How To Prepare A Safer Sample

- Remove direct identifiers such as names, emails, phone numbers, and addresses.
- Replace customer IDs with fake IDs.
- Keep the columns that matter for the business question.
- Include 500-5,000 representative rows if possible.
- Tell us which columns should not be used in analysis.
- Tell us whether any fields are future-looking, proxy scores, or manually created labels.

## What Archimedes Provides

Archimedes produces a first-pass action report:

- Summary of the business issue.
- Important segments.
- Risks and drivers.
- Suggested review queues.
- Operating rules.
- Data quality notes.

## What Archimedes Does Not Provide

Archimedes does not provide:

- Legal, medical, or financial advice.
- Fully automated decisions.
- A guarantee that the model is correct.
- A replacement for human review.
- A governed enterprise BI or compliance system.

## Plain-Language Disclaimer

Archimedes is designed to help teams understand messy operational data and decide what to review next. Outputs should be checked by a human before being used for customer, financial, operational, or strategic decisions.

