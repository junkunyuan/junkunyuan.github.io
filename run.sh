PROJECT_ROOT=$(dirname "$(realpath "$0")")
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo pwd: $(pwd)

trap 'echo "Exiting..."; exit 0' INT TERM

while true; do
    echo "=========================================="
    echo "Start running - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    python -m build_index
    python -m paper_reading_list.build_html
    
    sleep 3
done
