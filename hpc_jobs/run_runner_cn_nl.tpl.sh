#!/bin/bash
#SBATCH -J gp_runner_${RUNNER_ID}
#SBATCH -p cn_nl
#SBATCH -A chongbin_g1
#SBATCH --qos=chongbincnnl
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 04:00:00
#SBATCH -o gp_runner_${RUNNER_ID}_%j.out
#SBATCH -e gp_runner_${RUNNER_ID}_%j.err

export QUEUE_ROOT="$HOME/gp_queue"
export SHARE_ROOT="$HOME"
export RUNNER_ID="${RUNNER_ID}"
export APPTAINER_BIN="singularity"

cd $HOME/MARL_CGM
python -m hpc.runner
