set -e; mkdir -p ext/sk
B=https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches; M=https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches
for id in 1899585 1925299 1927964 1953632 1959846 1986691 1996436 2006363 2007448 2007721 2010085 2011166 2013725 2015213; do
  [ -s ext/sk/${id}_tracking.jsonl ] || curl -sSL -o ext/sk/${id}_tracking.jsonl $M/$id/${id}_tracking_extrapolated.jsonl
  curl -sSL -o ext/sk/${id}_match.json $B/$id/${id}_match.json
  curl -sSL -o ext/de_${id}.csv $B/$id/${id}_dynamic_events.csv
  echo got $id
done; echo FETCHDONE
