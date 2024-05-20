#! /bin/bash

#SBATCH --partition=NTU_Heart_CT
#SBATCH --nodes=1
#SBATCH --job-name=convert_hsinchu_232
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x/%j.out
#SBATCH --cpus-per-task=32

export HSU="/mnt/home/hsu.william9"
export SHARE="/mnt/share/NTU_Heart_CT/"
export CONTAINER_NAME="$HSU/workspace/dimage/nemo_monai.sqsh"
export SRC_MNT="$SHARE/usbD/RAW/batch2-2024-04-30/CCTA2022全年+2024.1.1-2024.4.24(232例):/workspace/src-232"
export DST_MNT="$SHARE/data:/workspace/data"
export PROJ_MNT="$HSU/workspace/project:/workspace/project"
export CONTAINER_MOUNT="${SRC_MNT},${DST_MNT},${PROJ_MNT}"

export PYTHON_CMD="
python /workspace/project/ConvertScripts/dcm2niix_HsinChu.py \
--data_root=/workspace/src-232 --out_dir=/workspace/data/image/HsinChu \
--buf_dir=/workspace/buf --err_dir=/workspace/data/err/HsinChu \
--dcm2niix=/workspace/project/ConvertScripts/lib/dcm2niix --num_workers=16
"

srun --container-image=$CONTAINER_NAME --container-mounts=$CONTAINER_MOUNT --container-writable $PYTHON_CMD