# Archimedes Analytics Setup Draft

This folder prepares a free GA4-to-Google-Sheets workflow. It is not active until the website gets a real GA4 measurement ID and the Google Sheet gets the Apps Script installed.

## Goal

Track early marketing signals without paid tools:

- Website visits
- Engaged visits
- Report example clicks
- Contact section views
- Email button clicks
- Top traffic source/channel
- Weekly scorecard rows in Google Sheets

## Privacy / Internal Traffic

- Do not track or store individual visitor IP addresses in the marketing sheet.
- Do not treat owner, developer, local, or agent testing traffic as real marketing demand.
- GA4 does not expose individual visitor IP addresses in normal reports.
- Use GA4 internal/developer traffic filters when possible, and keep manual notes in `internal_traffic_exclusions.csv`.

## Files

- `ga4-website-snippet.html` - tracking snippet template for the website.
- `apps-script/Code.gs` - Apps Script template that pulls GA4 weekly metrics into the tracker.
- `apps-script/appsscript.json` - Apps Script manifest scopes.

## Setup Order

1. Create a GA4 property for Archimedes.
2. Get the GA4 web measurement ID, like `G-XXXXXXXXXX`.
3. Add the website snippet after replacing the placeholder ID.
4. Confirm events appear in GA4 Realtime/DebugView.
5. Import the tracker CSV files into Google Sheets.
6. Open Google Sheets -> Extensions -> Apps Script.
7. Add `Code.gs` and `appsscript.json`.
8. Replace `GA_PROPERTY_ID` with the numeric GA4 property ID.
9. Run `runWeeklyMarketingPull` manually once.
10. After confirming the row looks right, create a weekly trigger.

## Events

Recommended event names:

- `arch_report_example_click`
- `arch_contact_section_view`
- `arch_email_click`
- `arch_demo_click`

Only `arch_email_click` means someone clicked the email link. It does not prove they sent an email. Actual email inquiries should still be logged manually in the `inquiry_log` tab unless the contact flow later becomes a real form submission.

## What Not To Automate Yet

- Do not bulk-email leads automatically.
- Do not log IP addresses into Sheets.
- Do not count internal testing as marketing traffic.
- Do not connect paid tools.
