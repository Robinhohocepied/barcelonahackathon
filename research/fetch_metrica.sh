#!/bin/bash
# Metrica Sports open sample data (3 anonymised games). Source: github.com/metrica-sports/sample-data
set -e
mkdir -p ext/metrica && cd ext/metrica
B=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for g in 1 2; do for f in RawEventsData RawTrackingData_Home_Team RawTrackingData_Away_Team; do
  curl -sSL -o Sample_Game_${g}_$f.csv $B/Sample_Game_$g/Sample_Game_${g}_$f.csv; done; done
for f in events.json metadata.xml tracking.txt; do curl -sSL -o Sample_Game_3_$f $B/Sample_Game_3/Sample_Game_3_$f; done
