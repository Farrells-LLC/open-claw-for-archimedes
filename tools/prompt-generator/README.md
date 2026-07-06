# Archimedes Prompt Generator Panel

This is a standalone prototype for a pre-report prompt builder.

It lets a user:

- Upload a CSV locally in the browser
- Inspect row count, columns, likely numeric fields, categorical fields, and binary outcome fields
- Describe what they want the report to focus on
- Generate a polished prompt to paste before creating an Archimedes report

The CSV never leaves the browser in this static prototype. It is intended to be folded into the live dashboard later.

## How To Use

Open `index.html` in a browser, upload a CSV, fill out the brief, then copy the generated prompt.

## Product Intent

This should eventually become a separate panel in the app, before report generation:

1. Upload or select dataset
2. Answer a few business-focus questions
3. Generate/edit the report instruction
4. Send that instruction into the normal report generation flow

