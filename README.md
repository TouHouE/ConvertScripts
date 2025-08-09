# Taipei-1 Usage Convert Scripts
## 1. Requirements
- pydicom
- nibabel
- pandas
- numpy
## 2. Converting .dcm to .nii.gz Files
- `dcm2niix_HsinChu.py`
  - This script has been **abandoned**.
- `dcm2niix_Taipei.py`
  - **Do not** use this script for decompression.
  - The dictionary should be like:
    ```
    /media/shard/data/ct_scan/raw/Taipei_2897/CGMH  <-- That is for argument `--data_root`
    ├── 0001  <-- here are the patient
    ├── 0003
    └── ......
    /media/shard/data/ct_scan/raw/Taipei_2897/ISP_annotation <-- That one should pass into `--isp_root`
    ├── 0001  # Almost like 502-CT
    ├── 0003
    └── ......
    ```
    - PS: To ignore parts that have already been processed, the `--ignore_path` should point to the `--meta_dir`.
    
  - The progress
    - A **Skip ISP** status indicates that the ISP folder does not contain any tissue labels.
## 3. Extract the centerline coordinates and names of each coronary artery and plaque.
```shell!
python store_each_plq.py ...
```
## 4. To Merge Duplicate Plaque Labels
```shell
python combine_multi_plque.py ... 
```

## PS
- The dcm2niix program just download from [dcm2niix](https://github.com/rordenlab/dcm2niix/releases)
- Cardiac Structure ID to Name
  1. Right Atrium
  2. Right Ventricle
  3. Left Atrium
  4. Left Ventricle
  5. MyocardiumLV
  6. Aorta
  7. Coronaries8
  8. Fat
  9. Bypass
  10. Plaque
 
