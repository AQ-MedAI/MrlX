#!/bin/bash

# Get script directory
QUICKSTART_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Show help information
show_help() {
    echo "=========================================="
    echo "  MrlX-DeepResearch Quick Start"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  bash quick_start.sh [main|sub]"
    echo ""
    echo "Arguments:"
    echo "  main  - Start Main Agent environment"
    echo "  sub   - Start Sub Agent environment"
    echo ""
    echo "Examples:"
    echo "  bash quick_start.sh main    # Start Main Agent"
    echo "  bash quick_start.sh sub     # Start Sub Agent"
    echo ""
    echo "For detailed instructions, see: README_QUICKSTART.md"
    echo "=========================================="
    echo ""
    echo "Before first use, ensure you have completed:"
    echo "0. Prepare model checkpoints (download and convert to torch_dist format)"
    echo "1. Get Sub Agent IP: hostname -i (in sub container)"
    echo "2. Copy template: cp env.example .env"
    echo "3. Edit .env file and fill in all required variables (including model paths)"
    echo "4. Copy .env file to both main and sub containers"
    echo ""
}

# Check arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Execute corresponding startup script based on argument
case "$1" in
    main)
        echo "Starting Main Agent..."
        bash "$QUICKSTART_ROOT/quick_start_main.sh"
        ;;
    sub)
        echo "Starting Sub Agent..."
        bash "$QUICKSTART_ROOT/quick_start_sub.sh"
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Error: Unknown argument '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac
