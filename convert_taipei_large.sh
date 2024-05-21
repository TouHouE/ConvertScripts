#! /bin/bash

#SBATCH --partition=NTU_Heart_CT
#SBATCH --job-name=convert_taipei_large
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x/%j.out
#SBATCH --cpus-per-task=32

export HSU="/mnt/home/hsu.william9"
export LOUIE="/mnt/home/l159753807"
export SHARE="/mnt/share/NTU_Heart_CT"
export CONTAINER_NAME="$HSU/workspace/dimage/nemo_monai.sqsh"

export SRC_MNT="$SHARE/usbC/all_ct:/workspace/src/ct"
export ISP_MNT="$SHARE/large_isp:/workspace/src/isp"
export DST_MNT="$SHARE/data:/workspace/data"
export PROJ_MNT="$HSU/workspace/project:/workspace/project"

export CONTAINER_MOUNT="$SRC_MNT,$DST_MNT,$PROJ_MNT,$PROJ_MNT"

export PYTHON_CMD="
python /workspace/project/ConvertScripts/dcm2niix_Taipei.py \
--data_root=/workspace/src/ct --isp_root=/workspace/src/isp \
--out_dir=/workspace/data/image/Taipei/LargeCT --mask_dir=/workspace/data/mask/Taipei/LargeCT
--buf_dir=/workspace/buf --err_dir=/workspace/data/err/Taipei/LargeCT --ignore_path=no \
--dcm2niix=/workspace/project/ConvertScripts/lib/dcm2niix --num_workers=16 \
--dst_root=/workspace/data --meta_dir=/workspace/data/meta/Taipei/LargeCT
"

srun --container-image=$CONTAINER_NAME --container-mounts=$CONTAINER_MOUNT --container-writable $PYTHON_CMD