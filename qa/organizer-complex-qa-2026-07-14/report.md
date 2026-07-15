# Archimedes Data Organizer Complex QA - 2026-07-14

Five realistic messy organizer scenarios were tested against a lightweight harness that mirrors the local organizer execution behavior: append, single-key outer join, renames, casts, date standardization, delimited-column parsing, dedupe, and row-count validation.

Important limitation: this did not test the AI planner or authenticated FastAPI endpoints because this shell lacks the app dependencies and live auth environment. It tests what happens when realistic plans are executed.

Overall: 1 passed, 0 passed with warnings, 4 failed.

## case1_ecom_multikey - Ecommerce orders need customer_id and sku enrichment

Status: fail
Output: 9 rows x 11 columns
Built-in-style validation warnings:
- Fallback append used for 1 file(s): [{'file': 'products.csv', 'reason': "'customer_id'", 'action': 'fallback_append'}]
- Merge key 'customer_id' not present in every file
What went wrong:
- No fallback append should be needed for multi-file enrichment: ['Fallback append used for 1 file(s): [{\'file\': \'products.csv\', \'reason\': "\'customer_id\'", \'action\': \'fallback_append\'}]', "Merge key 'customer_id' not present in every file"]
- Every order row should retain product category: some order rows missing category

## case2_monthly_stack_currency - Monthly sales exports with renamed headers, dates, currency, and percents

Status: fail
Output: 4 rows x 5 columns
Built-in-style validation warnings: none
What went wrong:
- Revenue should not be lost during currency cleanup: null revenue count 2
- Discount percent should parse from percent strings: null discount count 2

## case3_support_one_to_many - Customer health joined to support tickets creates grain change

Status: fail
Output: 6 rows x 5 columns
Built-in-style validation warnings: none
What went wrong:
- Organizer should warn when one-to-many join changes customer-level grain: output rows 6; built-in warnings []

## case4_leading_zero_ids - Customer IDs with leading zeros should match numeric-looking IDs safely

Status: fail
Output: 6 rows x 5 columns
Built-in-style validation warnings: none
What went wrong:
- IDs should match despite leading-zero formatting differences: got 6 rows; matched orders 3
- Validation should flag low/no match rate: built-in warnings []

## case5_semicolon_parse_append - Semicolon-delimited export gets split and appended to clean CRM rows

Status: pass
Output: 4 rows x 4 columns
Built-in-style validation warnings: none
What worked: all custom assertions passed.

## Main Findings

- The organizer can handle a simple delimited-column parse plus append path.
- The current single merge-key model is not enough for common ecommerce enrichment, where orders often need customer_id and sku joins in the same workflow.
- Currency and percent cleanup are not strong enough if values contain symbols like `$` or `%`; simple numeric casting can null out useful values.
- One-to-many joins can silently change the grain of the dataset from customer-level to ticket/order-level without enough warning.
- ID normalization needs stronger handling for leading zeros and numeric-looking IDs; otherwise joins can produce unmatched duplicate entities while still passing row-count checks.

## Recommended Fixes

1. Support multi-step joins with different keys, not just one global merge_key.
2. Add match-rate diagnostics for every join: matched rows, left-only rows, right-only rows, duplicate keys, and row multiplier.
3. Add grain detection and warnings for one-to-many and many-to-many joins.
4. Add currency/percent parsers before numeric casts.
5. Add ID-safe normalization that preserves leading zeros and warns before coercing identifiers.
6. Fail loudly when a file falls back from join to append, instead of letting a mixed output look successful.