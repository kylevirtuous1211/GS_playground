#!/usr/bin/env bash
# Event stream for a running plan: stage lines, failures, verdicts, and stalls.
# Each argument is a runner log; the watch ends when every log has an "exit N" line.
#   bash tools/watch_plan.sh <runner.log> [<runner.log> ...]
# A log that does not exist yet is waited for; a log not touched for 15 min is reported once.
set -u
STALE_S=900
declare -A seen stale_reported
while true; do
    all_done=1
    for log in "$@"; do
        [ -f "$log" ] || { all_done=0; continue; }
        n=$(wc -l < "$log")
        if [ "${seen[$log]:-0}" -lt "$n" ]; then
            tail -n +"$(( ${seen[$log]:-0} + 1 ))" "$log" | head -n "$(( n - ${seen[$log]:-0} ))" |
                grep -E '^== |FAILED|Traceback|Error|sanity|passes|verdict|compatible|CHANGED|exit [0-9]' |
                sed "s|^|[$(basename "$(dirname "$log")")] |" | cut -c1-240
            seen[$log]=$n
            stale_reported[$log]=0
        fi
        grep -q '^exit [0-9]' "$log" && continue
        all_done=0
        # the newest file under the log's directory shows whether anything is still being written
        newest=$(find "$(dirname "$log")" -type f -newermt "-${STALE_S} seconds" -print -quit 2>/dev/null)
        if [ -z "$newest" ] && [ "${stale_reported[$log]:-0}" = 0 ]; then
            echo "[$(basename "$(dirname "$log")")] STALE: nothing written for 15 min"
            stale_reported[$log]=1
        fi
    done
    [ "$all_done" = 1 ] && { echo "ALL RUNNERS FINISHED"; break; }
    sleep 60
done
