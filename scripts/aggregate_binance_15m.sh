#!/bin/sh
set -eu

root="${1:-/Users/akshay/Documents/HIST_DATA/BTC:USDT/M1}"
output="${2:-research_artifacts/market_timeframe/BTCUSDT_15min.csv}"

mkdir -p "$(dirname "$output")"
{
  printf 'timestamp,open,high,low,close,volume\n'
  for archive in "$root"/*.zip; do
    unzip -p "$archive"
  done | awk -F, '
    function emit() {
      if (have) printf "%.0f,%.10g,%.10g,%.10g,%.10g,%.10g\n", bucket, open, high, low, cls, volume
    }
    {
      timestamp = $1
      if (timestamp > 1000000000000000) timestamp = int(timestamp / 1000)
      next_bucket = int(timestamp / 900000 + 1) * 900000
      if (!have || next_bucket != bucket) {
        emit()
        bucket = next_bucket
        open = $2
        high = $3
        low = $4
        cls = $5
        volume = $6
        have = 1
      } else {
        if ($3 > high) high = $3
        if ($4 < low) low = $4
        cls = $5
        volume += $6
      }
    }
    END { emit() }
  '
} > "$output"
