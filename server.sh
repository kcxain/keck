#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J cuda_server              # The job name
#SBATCH -o cuda_server.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e cuda_server.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw-80g               # Submit to 'r8nv-gpu-hw' Partitiion
#SBATCH -t 1-00:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:8                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-normal             # Request QOS Type

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
#SBATCH --cpus-per-task=128            # Request K cores
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### #SBATCH --nodelist=r8a100-d04          # Request a specific list of hosts 
### #SBATCH --constraint="A30|A100"      # Request GPU Type: A30 or A100_40GB
###

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "$(df -h | grep -v tmpfs)"

#- Important settings!!!
# 1. Prevents RDMA resource exhaustion errors:
ulimit -l unlimited
# 2. Prevents virtual memory exhaustion errors, which are critical
#    when loading Large Language Models (LLMs):
ulimit -v unlimited
# 3. Increases the maximum number of open file descriptors to avoid
#    issues with too many concurrent connections or file accesses:
ulimit -n 65535
# 4. Raises the maximum number of user processes to support
#    large-scale parallel workloads:
ulimit -u 4125556

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

#- Other

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- WARNING! DO NOT MODIFY your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES}"                         # which GPUs
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid $SLURM_JOB_ID -dd | awk '/IDX/ {print $2, $4}'

ray stop

ray start --head

# 对于集群用这个：
# export CUDA_HOME=/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 32

echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
# This will overwrite any existing atop logs from previous runs.
# WARNING: If your program times out or is terminated by scancel,
#          the above script part might not execute correctly.
