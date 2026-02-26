#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [--no-clean]"
    echo "  --no-clean    Do not delete old log and run files."
    exit 1
}

# Parse command-line arguments
CLEAN=true
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-clean)
            CLEAN=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Function to submit a job
submit_job() {
    local config_file=$1
    local config_number=$(basename "$config_file" .json | sed 's/config_//')
    local job_name="sim_${config_number}"

    cmd="python run_simulation.py --config $config_file"

    tmp_job_script=$(mktemp)
    cat << EOF > "$tmp_job_script"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=highmem
#SBATCH --mem=64G

$cmd
EOF

    sbatch "$tmp_job_script"
    rm "$tmp_job_script"

    echo "Submitted job: $cmd"
}

# Ensure the logs directory exists
mkdir -p logs

# Perform cleanup if CLEAN is true
if [ "$CLEAN" = true ]; then
    echo "Cleaning old log files and run files..."
    rm -f *.out *.err
    rm -f logs/*.out logs/*.err
    #rm -f runs/*.h5
    #find runs -name '*.h5' -delete
    echo "Cleanup completed."
else
    echo "Skipping cleanup as per user request."
fi

# Generate config files
python generate_configs.py

# Submit jobs for all config files
for config_file in configs/*.json; do
    submit_job "$config_file"
done