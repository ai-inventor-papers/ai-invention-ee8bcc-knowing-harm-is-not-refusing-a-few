#!/bin/bash
while kill -0 2483 2>/dev/null; do
  tail -1 logs/run_full2.log
  echo "checkpoints=$(ls caches/per_model/*.json 2>/dev/null | wc -l)"
  sleep 60
done
echo "MONITOR: full stage ended"
