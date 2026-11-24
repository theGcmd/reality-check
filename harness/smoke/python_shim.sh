#!/bin/bash
# Smoke shim: run only the FIRST python invocation of a repo script, 1 epoch, our data path.
M=/home/claude/reality-check/smoke/.done_$SMOKE_NAME
[ -f "$M" ] && exit 0
touch "$M"
LOG=/home/claude/reality-check/smoke/$SMOKE_NAME.log
echo "CMD: python $* --train_epochs 1 --root_path $ETT_ROOT" > "$LOG"
S=$(date +%s)
CPU_REDIRECT=1 WANDB_MODE=disabled timeout ${SMOKE_TIMEOUT:-900} /home/claude/reality-check/tslenv/bin/python "$@" --train_epochs 1 --root_path "$ETT_ROOT" >> "$LOG" 2>&1
RC=$?
echo "SMOKE_RC=$RC SECONDS=$(( $(date +%s) - S ))" >> "$LOG"
exit 0
