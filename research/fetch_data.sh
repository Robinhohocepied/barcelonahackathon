#!/bin/bash
# SkillCorner open data (MIT): 6 matches of tracking (Git LFS, ~90 MB each) + possession events.
# Run from this folder. Creates ext/sk/ and ext/de_<id>.csv
set -e
mkdir -p ext/sk
for id in 1886347 1996435 2017461 2016236 2006229 1874553; do
  curl -sSL -o ext/sk/${id}_tracking.jsonl "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/$id/${id}_tracking_extrapolated.jsonl"
  curl -sSL -o ext/sk/${id}_match.json "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/$id/${id}_match.json"
  curl -sSL -o ext/de_${id}.csv "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/$id/${id}_dynamic_events.csv"
done
