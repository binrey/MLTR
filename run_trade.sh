#!/bin/bash

# Script to run volprof BTCUSDT with optional debug flag
# Usage: ./run_volprof.sh [--debug]

DEBUG=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 run.py bybit configs/volprof/BTCUSDT.py"
if [ "$DEBUG" -eq 1 ]; then
    CMD="$CMD --debug"
    echo "Running in debug mode"
fi

# Execute command
echo "Executing: $CMD"
$CMD