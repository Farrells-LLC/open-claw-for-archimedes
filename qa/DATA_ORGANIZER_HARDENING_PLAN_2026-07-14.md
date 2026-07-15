# Archimedes Data Organizer Hardening Plan - 2026-07-14

This plan is for making the data organizer live up to the promise before it is used heavily in outreach or customer demos.

## Current Evidence

Prior testing shows the system has real working paths, but it should not be treated as fully hardened yet.

What already passed:

- 5/5 large structured dataset workflows completed end-to-end in the July 5 overnight QA.
- Upload/analyze, target-aware report generation, clarify endpoint, and train/test model workflows completed on 5,000-row datasets.
- A 9,000-row, 58-column clinical readmission retest completed after prompt-size handling was improved.
- A live organizer join smoke test passed for 3 split files joined on `customer_id`, preserving 30 distinct customer records and producing 6 merged columns.
- Ecommerce and SaaS demo report workflows generated useful public-facing examples.

Known gaps or risks:

- Earlier large report runs failed around the LLM prompt limit before sampling/truncation was added.
- Pre-target task detection was sometimes wrong, such as churn/credit being detected as regression and demand forecasting being detected as classification.
- Synthetic datasets included leakage/proxy fields, so the product needs leakage warnings and exclusion prompts.
- Join testing covered one clean happy path, not messy real-world files.
- The organizer still needs harsher tests for bad headers, duplicate keys, mismatched row counts, malformed dates, currency symbols, missing values, encoding issues, and accidental row loss.

## Definition Of Ready

The data organizer is ready to advertise more confidently when it can handle these without silent failure:

- Upload one or more CSV files.
- Detect schema, row count, column count, missingness, duplicate headers, and data types.
- Suggest whether to stack, join, clean, dedupe, or leave files separate.
- Ask for clarification when merge keys or target columns are ambiguous.
- Preserve rows unless the user explicitly chooses a row-reducing operation.
- Warn before destructive cleanup.
- Produce a clear before/after summary.
- Export the cleaned/organized file and a human-readable change log.
- Feed the organized data into the report workflow without losing context.

## Test Matrix

### Single-File Cleanup

Test cases:

- Clean CSV with simple numeric and categorical columns.
- Messy headers with spaces, punctuation, duplicate names, and casing differences.
- Missing values in numeric, categorical, date, and ID columns.
- Currency and percent strings such as `$1,204.55`, `18%`, and `(400.00)`.
- Date formats mixed across rows.
- Boolean values expressed as yes/no, TRUE/FALSE, 1/0, and blank.
- High-cardinality text columns.
- Very wide files with 100+ columns.
- Larger files with 25k, 100k, and 250k rows if infrastructure allows.

Pass criteria:

- The organizer explains what it changed.
- It does not silently coerce important IDs into numbers.
- It preserves the original row count unless the selected operation should reduce rows.
- It flags columns that could not be safely parsed.

### Multi-File Stack

Test cases:

- Same columns in same order.
- Same columns in different order.
- One file missing optional columns.
- One file has extra columns.
- Header synonyms such as `customer_id`, `Customer ID`, and `cust id`.
- Conflicting data types for the same column.

Pass criteria:

- Row counts add up correctly.
- Column mapping is shown before export.
- Missing/extra columns are explained.
- Source filename column can be added for traceability.

### Multi-File Join

Test cases:

- Clean one-to-one join on a shared ID.
- One-to-many join.
- Many-to-many join that could explode row count.
- Missing join keys in one file.
- Duplicate keys in one file.
- Keys with leading zeros, whitespace, mixed casing, and numeric/string mismatch.
- Partial overlap where only some records match.
- Files that should not be joined even though they share a similar column name.

Pass criteria:

- Organizer identifies likely join keys and asks for confirmation.
- It warns on duplicate keys and many-to-many row explosion risk.
- It reports matched, unmatched-left, unmatched-right, and output row counts.
- It never treats join input rows as if they should simply sum.
- It preserves IDs exactly, including leading zeros.

### Report Workflow After Organizing

Test cases:

- Organized file feeds into report generation.
- Organized file feeds into model training with explicit target selection.
- User changes target after organizer output.
- Report prompt receives sampled/summarized data instead of entire oversized raw data.

Pass criteria:

- No prompt-limit failure on large/wide files.
- Report states whether the data was sampled or summarized.
- Target/task suggestions refresh after target selection.
- Leakage/proxy columns are flagged before training.

### Failure Handling

Test cases:

- Bad CSV quoting.
- Non-UTF-8 encoding.
- Empty file.
- File with only headers.
- File with no headers.
- Uploaded Excel file if supported.
- Network interruption or long-running report timeout.

Pass criteria:

- User sees a useful error message.
- No partial output is presented as complete.
- Logs capture the failure reason.
- Retry path is clear.

## First Hardening Sprint

1. Recreate the existing clean join test as an automated regression test.
2. Add messy join fixtures: duplicate keys, missing keys, partial overlap, and many-to-many risk.
3. Add stack fixtures for same/different column order and missing columns.
4. Add single-file cleanup fixtures for headers, dates, currency, percent, IDs, and blanks.
5. Add row/column accounting assertions to every organizer operation.
6. Add before/after change log assertions.
7. Add prompt-size regression test using a wide/large file.
8. Add leakage/proxy warning checks before model training.
9. Run the organized output through report generation for ecommerce and SaaS demos.
10. Save screenshots or artifacts for every public demo claim.

## Do Not Claim Yet

Until the hardening sprint passes, avoid claims like:

- Handles any messy spreadsheet.
- Fully automated data cleaning.
- Production-ready data prep.
- No manual review needed.

Safer claim:

- Archimedes is being tested as a lightweight way to turn messy business exports into focused action reports, with human review before decisions.

## Immediate Next Move

Build or run the automated organizer test matrix before increasing outreach volume. The marketing promise should stay slightly behind verified product behavior.
