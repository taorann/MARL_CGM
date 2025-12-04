#!/bin/bash
#SBATCH -p cn_nl
#SBATCH -A chongbin_g1
#SBATCH --qos=chongbincnnl
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 04:00:00
#SBATCH -o logs/gp_runner_%x_%j.out   # %x=job name, %j=job id
#SBATCH -e logs/gp_runner_%x_%j.err

# 注意：这里不要再写 #SBATCH -J，留给 sbatch 命令行指定

# 保证 logs 目录存在
mkdir -p "$HOME/MARL_CGM/logs"

# 配置 runner 环境变量
export QUEUE_ROOT="$HOME/gp_queue"
export RUNNER_ID="${RUNNER_ID}"
export SHARE_ROOT="$HOME"
export APPTAINER_BIN="singularity"

cd "$HOME/MARL_CGM"

echo "[runner] NODE: $(hostname)"
echo "[runner] QUEUE_ROOT=$QUEUE_ROOT"
echo "[runner] RUNNER_ID=$RUNNER_ID"

python -m hpc.runner
