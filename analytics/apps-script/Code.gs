const CONFIG = {
  GA_PROPERTY_ID: 'REPLACE_WITH_NUMERIC_GA4_PROPERTY_ID',
  TIMEZONE: 'America/New_York',
  SCORECARD_SHEET_NAMES: ['weekly_scorecard', 'Weekly Scorecard'],
  EVENTS: [
    'arch_contact_section_view',
    'arch_email_click',
    'arch_report_example_click',
    'arch_demo_click'
  ]
};

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Archimedes')
    .addItem('Pull weekly GA4 metrics', 'runWeeklyMarketingPull')
    .addToUi();
}

function runWeeklyMarketingPull() {
  if (!CONFIG.GA_PROPERTY_ID || CONFIG.GA_PROPERTY_ID.indexOf('REPLACE_') === 0) {
    throw new Error('Set CONFIG.GA_PROPERTY_ID to the numeric GA4 property ID first.');
  }

  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = findSheet_(spreadsheet, CONFIG.SCORECARD_SHEET_NAMES);
  const range = getLastFullWeekRange_();

  const overview = fetchRunReport_({
    dateRanges: [{ startDate: range.startDate, endDate: range.endDate }],
    metrics: [
      { name: 'sessions' },
      { name: 'totalUsers' },
      { name: 'engagedSessions' },
      { name: 'screenPageViews' }
    ]
  });

  const eventCounts = fetchEventCounts_(range);
  const topChannel = fetchTopChannel_(range);

  const websiteVisits = getMetric_(overview, 'sessions');
  const engagedVisits = getMetric_(overview, 'engagedSessions');
  const contactSectionViews = eventCounts.arch_contact_section_view || 0;
  const emailClicks = eventCounts.arch_email_click || 0;
  const reportClicks = eventCounts.arch_report_example_click || 0;

  sheet.appendRow([
    range.weekStarting,
    websiteVisits,
    engagedVisits,
    contactSectionViews,
    '',
    '',
    '',
    '',
    topChannel,
    `GA4 pull: ${reportClicks} report clicks, ${emailClicks} email clicks.`,
    'Review inquiries, update outreach, and compare against prior week.',
    'Exclude owner/dev/agent testing traffic during review.'
  ]);
}

function fetchEventCounts_(range) {
  const response = fetchRunReport_({
    dateRanges: [{ startDate: range.startDate, endDate: range.endDate }],
    dimensions: [{ name: 'eventName' }],
    metrics: [{ name: 'eventCount' }],
    dimensionFilter: {
      filter: {
        fieldName: 'eventName',
        inListFilter: { values: CONFIG.EVENTS }
      }
    }
  });

  const counts = {};
  (response.rows || []).forEach(row => {
    counts[row.dimensionValues[0].value] = Number(row.metricValues[0].value || 0);
  });
  return counts;
}

function fetchTopChannel_(range) {
  const response = fetchRunReport_({
    dateRanges: [{ startDate: range.startDate, endDate: range.endDate }],
    dimensions: [{ name: 'sessionDefaultChannelGroup' }],
    metrics: [{ name: 'sessions' }],
    orderBys: [{ metric: { metricName: 'sessions' }, desc: true }],
    limit: '1'
  });

  if (!response.rows || !response.rows.length) return '';
  const row = response.rows[0];
  return `${row.dimensionValues[0].value} (${row.metricValues[0].value} sessions)`;
}

function fetchRunReport_(body) {
  const url = `https://analyticsdata.googleapis.com/v1beta/properties/${CONFIG.GA_PROPERTY_ID}:runReport`;
  const response = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: `Bearer ${ScriptApp.getOAuthToken()}`
    },
    payload: JSON.stringify(body),
    muteHttpExceptions: true
  });

  const status = response.getResponseCode();
  const text = response.getContentText();
  if (status < 200 || status >= 300) {
    throw new Error(`GA4 Data API request failed (${status}): ${text}`);
  }
  return JSON.parse(text);
}

function getMetric_(response, metricName) {
  if (!response.rows || !response.rows.length) return 0;
  const index = response.metricHeaders.findIndex(header => header.name === metricName);
  if (index < 0) return 0;
  return Number(response.rows[0].metricValues[index].value || 0);
}

function getLastFullWeekRange_() {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const day = today.getDay();
  const daysSinceMonday = (day + 6) % 7;

  const thisMonday = new Date(today);
  thisMonday.setDate(today.getDate() - daysSinceMonday);

  const lastMonday = new Date(thisMonday);
  lastMonday.setDate(thisMonday.getDate() - 7);

  const lastSunday = new Date(thisMonday);
  lastSunday.setDate(thisMonday.getDate() - 1);

  return {
    weekStarting: formatDate_(lastMonday),
    startDate: formatDate_(lastMonday),
    endDate: formatDate_(lastSunday)
  };
}

function formatDate_(date) {
  return Utilities.formatDate(date, CONFIG.TIMEZONE, 'yyyy-MM-dd');
}

function findSheet_(spreadsheet, names) {
  for (const name of names) {
    const sheet = spreadsheet.getSheetByName(name);
    if (sheet) return sheet;
  }
  throw new Error(`Could not find scorecard sheet. Expected one of: ${names.join(', ')}`);
}
