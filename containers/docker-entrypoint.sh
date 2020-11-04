#!/bin/bash
set -e

if [ "$1" = 'sltp' ]; then
    # Pass only the relevant arguments to the main SLTP script
    exec /root/projects/sltp/experiments/run.py "${@:2}"
else
    exec "$@"
fi

