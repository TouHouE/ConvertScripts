#! /bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=convert_hsinchu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
export HSU="/mnt/home/hsu.william9"
export LOUIE="/mnt/home/l159753807"
export JSON_LIST_IN_CONTAINER="/workspace/data/ct.json"
export DATA_IN_CONTAINER="/workspace/data/"
export CONTAINER_NAME="$HSU/workspace/dimage/nemo_monai.sqsh"
export SRC_MNT="$LOUIE/usbC:/workspace/src"
export ISP_MNT="$LOUIE/usbC/CT標註result:/workspace/src/isp"
export DST_MNT="$HSU/data:/workspace/data"
export PROJ_MNT="$HSU/project:/workspace/project"
export CONTAINER_MOUNT="$SRC_MNT,$DST_MNT,$PROJ_MNT"

export PYTHON_CMD="
python /workspace/project/ConvertScripts/dcm2niix_Taipei.py --large_ct \
--data_root=/workspace/src --isp_root=/workspace/src/isp \
--out_dir=/workspace/data/image/Taipei/LargeCT --buf_dir=/workspace/buf \
--err_dir=/workspace/data/err/Taipei/LargeCT \
--dcm2niix=/workspace/project/ConvertScripts/lib/dcm2niix.exe --num_workers=16 \
--dst_root=/workspace/data --
"

srun --container-name=$CONTAINER_NAME --container-mounts=$CONTAINER_MOUNT --container-writable bash -c $PYTHON_CMD