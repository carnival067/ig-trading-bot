# Free News Safety Layer

The `GUARDED_AUTO` and `PROFESSIONAL` entry paths support a restrictive-only
news overlay using:

- Financial Modeling Prep for the economic calendar and forex headlines.
- Marketaux for financial headlines and sentiment.
- Optional GDELT DOC API for geopolitical headlines.

Configure:

```text
FMP_API_KEY=
MARKETAUX_API_KEY=
ENABLE_NEWS_FILTER=true
NEWS_CHECK_INTERVAL_MINUTES=10
NEWS_BLOCK_BEFORE_HIGH_IMPACT_MINUTES=30
NEWS_BLOCK_AFTER_HIGH_IMPACT_MINUTES=45
ENABLE_GDELT_BACKUP=false
```

The layer is evaluated only after a strategy signal passes the existing entry
gate and central risk engine. It cannot create a signal, increase size, disable
a halt, or close a position.

Decision outputs:

- `news_risk_score`
- `news_action`
- `matched_news_headlines`
- `matched_calendar_events`
- `reason`

Actions are monotonic:

- `ALLOW_NORMAL`: no change.
- `REQUIRE_EXTRA_CONFIRMATION`: reject unless the strategy supplied an
  independent `extra_confirmation`.
- `REDUCE_SIZE`: halve the already risk-approved size.
- `BLOCK_TRADE`: reject the new entry.

When enabled, an unavailable FMP economic calendar blocks new entries. News
refreshes never alter existing IG positions, stops, or limits.
