#!/bin/bash
#SBATCH -J gp_runner_0
#SBATCH -p cn_nl
#SBATCH -A chongbin_g1
#SBATCH --qos=chongbincnnl
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 04:00:00
#SBATCH -o gp_runner_0_%j.out
#SBATCH -e gp_runner_0_%j.err

# 配置 runner 所需环境变量
export QUEUE_ROOT="$HOME/gp_queue"
export RUNNER_ID="0"
export SHARE_ROOT="$HOME"              # 简化：把 $HOME 挂到容器 /mnt/share
export APPTAINER_BIN="singularity"

cd $HOME/MARL_CGM

echo "[runner] NODE: \$(hostname)"
echo "[runner] QUEUE_ROOT=\$QUEUE_ROOT"
echo "[runner] RUNNER_ID=\$RUNNER_ID"

# 常驻循环，直到时间用完或你取消 job
python -m hpc.runner
