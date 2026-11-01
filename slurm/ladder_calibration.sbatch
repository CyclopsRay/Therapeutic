#!/usr/bin/env bash
#SBATCH --job-name=ladder-cal
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/ladder-cal-%A_%a.out
set -euo pipefail
# E9 + ladder calibration/power (PLAN.md Phase 6; design in report/10_30_record.md 2026-10-26).
# Usage: sbatch [--array=...] slurm/ladder_calibration.sbatch <subcommand> [args...]
#   Array tasks may pass @NET@ / @START@ / @STOP@ placeholders, filled from NETS and CHUNK below.
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
NETS=(string_functional700 signaling_tf)
i=${SLURM_ARRAY_TASK_ID:-0}; PER=${PER:-1}; CHUNK=${CHUNK:-100}
NET=${NETS[$((i / PER))]}; START=$(( (i % PER) * CHUNK )); STOP=$(( START + CHUNK ))
args=(); for a in "$@"; do a=${a//@NET@/$NET}; a=${a//@START@/$START}; a=${a//@STOP@/$STOP}; args+=("$a"); done
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
echo "LADDER_OUT=${LADDER_OUT:-default} LADDER_R=${LADDER_R:-39} task=$i args=${args[*]}"
cd /home/yuqi.lei/Therapeutic/scripts
"${ENV_ROOT}/bin/python" ladder_calibration.py "${args[@]}"
date --iso-8601=seconds
