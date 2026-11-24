#!/bin/bash
export ETT_ROOT=/home/claude/reality-check/data/ett/ETT-small/
export PATH=/home/claude/reality-check/smoke/shim:$PATH
R=/home/claude/reality-check/repos
while read name dir script args; do
  [ -z "$name" ] && continue
  export SMOKE_NAME=$name
  rm -f /home/claude/reality-check/smoke/.done_$name
  cd $R/$dir || continue
  mkdir -p logs/LongForecasting/etth1 logs/LongForecasting logs dataset
  ln -sfn $ETT_ROOT dataset/ETT-small 2>/dev/null
  echo "$(date +%T) START $name"
  bash $script $args > /home/claude/reality-check/smoke/$name.script.out 2>&1
  echo "$(date +%T) END $name $(tail -1 /home/claude/reality-check/smoke/$name.log 2>/dev/null)"
done < ${LIST:-/home/claude/reality-check/smoke/list.txt}
echo ALLDONE
