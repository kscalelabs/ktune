#!/bin/bash
# Source conda initialization
source ~/miniconda3/etc/profile.d/conda.sh

# Function to kill all kos-sim related processes
kill_kos_sim() {
  echo "Killing all kos-sim related processes..."
  # Kill any existing kos-sim processes
  pkill -f "kos-sim" || true
  pkill -f "kos_sim.server" || true
  
  # Give processes time to terminate
  sleep 1
  
  # Force kill if any still running
  pkill -9 -f "kos-sim" || true
  pkill -9 -f "kos_sim.server" || true
}

conda activate pykos3p11

# Define parameter lists
frictionlosses=(10 5 1 0.5 0.1 0.01 0.001 0.0001)
dampings=(10 5 1 0.5 0.1 0.01 0.001 0.0001)

# Iterate over every combination
for fl in "${frictionlosses[@]}"; do
  for dp in "${dampings[@]}"; do
    echo "Running simulation with frictionloss=${fl}, damping=${dp}"
    
    # Run the Python CLI with current params
    python cli.py --frictionloss "$fl" --damping "$dp"

    # Launch kos-sim in background, logging to a unique file
    echo "Starting kos-sim..."
    LOGFILE="kos-sim.log"
    kos-sim kbot-v2 --host 0.0.0.0 --suspended --no-cache >> "$LOGFILE" 2>&1 &
    SIM_PID=$!
    echo "kos-sim started (PID $SIM_PID), logging to $LOGFILE"


    echo "CHIRP NEXT"
    python chirp_kpkd_sweep.py --sim
    echo "STEP NEXT"
    python step_kpkd_sweep.py  --sim

    # Tear down
    echo "Killing kos-sim..."
    kill_kos_sim
    echo "Finished run for fl=${fl}, d=${d}"
    echo "---------------------------------------"
  done
done

echo "All simulations completed."