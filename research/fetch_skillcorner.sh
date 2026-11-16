#!/bin/bash
# SkillCorner open data (MIT): tracking (Git LFS) + match + dynamic events for 6 matches -> ext/sk/, ext/de_<id>.csv
set -e; mkdir -p ext/sk
B=https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches; M=https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches
for id in 1886347 1996435 2017461 2016236 2006229 1874553; do
  [ -s ext/sk/${id}_tracking.jsonl ] || curl -sSL -o ext/sk/${id}_tracking.jsonl $M/$id/${id}_tracking_extrapolated.jsonl
  curl -sSL -o ext/sk/${id}_match.json $B/$id/${id}_match.json
  curl -sSL -o ext/de_${id}.csv $B/$id/${id}_dynamic_events.csv
done
