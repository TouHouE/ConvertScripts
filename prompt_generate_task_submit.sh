#! /bin/bash

#SBATCH --partition=NTU_Heart_CT
#SBATCH --nodes=1
#SBATCH --job-name=pretrain_prompt
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x/%j.out
#SBATCH --cpus-per-task=4

export HSU="/mnt/home/hsu.william9"
export SHARE="/mnt/share/NTU_Heart_CT/"
export CONTAINER_NAME="$HSU/workspace/dimage/nemo_monai.sqsh"

export TXT_MNT="$SHARE/usbB/HsinChu/text:/workspace/text"
export CT_MNT="$SHARE/data/image/HsinChu:/workspace/ct"
export OUT_MNT="$SHARE/data:/workspace/data"
export PROJ_MNT="$HSU/workspace/project:/workspace/project"

export CONTAINER_MOUNT="${TXT_MNT},${CT_MNT},${OUT_MNT},${PROJ_MNT}"
export script="/workspace/project/ConvertScripts/prompt_generate_scripts.sh"
export PYTHON_CMD="

"

srun --container-image=$CONTAINER_NAME --container-mounts=$CONTAINER_MOUNT --container-writable bash $script