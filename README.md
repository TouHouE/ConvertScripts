# Taipei-1 Usage Convert Scripts
## 1. Requirements
- pydicom
- nibabel
- pandas
- numpy
## 2. Using to convert *.dcm file to *.nii.gz file
- `dcm2niix_HsinChu.py`
  - This script is using to convert *.dcm file that come from HsinChu Hospital
  - You might notice that there is no extension name in the raw file, and we also can't read each file directly; fortunately, `dcm2niix.exe` is still working!
  - The dictionary from HsinChu be like:
    ```
    /mnt/HsinChu
    ├── batch1 <-- the data_root must indicate this location
    │   ├──S000001  <-- The dictionary at this layer is meaning a patient
    │   ├──S000010
    │   └──......
    │       
    └── batch2
        ├── 2023.01-2023.12.... <-- the data_root must indicate this location
        │   ├── S000001  <-- therefore, this meaining patient
        │   ├── S000010
        │   └── ......
        ├── 2024.01-2024.03.... <-- this one also indicate as data_root
        │   └── ......
        └── .....
    ```
    - The batch1 is already uploaded on doctor's dropbox.
  - The storage structure will same as original
    - ex: \<the path for patient>/\<series number>/\<instance uid>/\<cardiac phase>/\<name generated by dcm2niix.exe>.nii.gz
  - Example:
    
      Follow the example dictionary structure, suppose we want to convert `/mnt/HsinChu/batch2/2023.01-2023.12`, and the target dictionary called `/mnt/hsinchu_dst`, the `dcm2niix.exe` place at `/mnt/lib/dcm2niix.exe`, then we simply named `out`, `buf`, `err` for output, buffer, error storage dictionary, and using 10 process, the command will look like: 
    ```bash=
    python dcm2niix_HsinChu.py --data_root=/mnt/HsinChu/batch2/2023.01-2023.12 --out_dir=/mnt/hsinchu_dst/out --buf_dir=/mnt/hsinchu_dst/buf --err_dir=/mnt/hsinchu_dst/err --num_workers=10 --dcm2niix=/mnt/lib/dcm2niix.exe
    ```
- `dcm2niix_Taipei.py`
  - This script is using to convert *.dcm file that come from Taipei Hospital
  - The dictionary from Taipei be like:
    - The 502-dicom:
      ```
      /mnt/usbB/'502-CT(dicom-format)  <-- That is for argument --data_root
      ├── 0001  <-- here are the patient
      ├── 0003
      └── ......
      /mnt/usbB/'CCTA Result' <-- That one should pass into --isp_root
      ├── 0001  # Almost like 502-CT
      ├── 0003
      └── ......
      ```
      - PS: Because in this part got some chaos
        1. I already convert around 4 / 5 of raw data(the *.dcm)
        2. But I just put around 300 patient on doctor's dropbox
        3. For ignore already processed part, the `--meta_dir` should point to the /mnt/usbA/502CT/meta 
    - But, for 2500+ dicom, that got different:
      ```
        /mnt/usbC 
        ├── CGMH  <-- That dictionary is for argument --data_root
        │   ├── 1557.zip  <-- I suggest unzip after uploaded, send single file should faster than single dictionary.
        │   ├── 0003
        │   └── ......
        ├── CMUH  <-- That is for argument --data_root
        │   ├── 0001  <-- here are the patient
        │   ├── 0003
        │   └── ......
        ├── MMH
        │   └── ......
        ├── MMH
        │   └── ......
        ├── ......
        └── CT標註result  <-- That one is --isp_root
        ```
  - The progress
    - if the status show **Skip ISP** it represent the isp folder didn't contain any tissue labels.
## 3. Extract each coronary artery centerline coordinate and name, also plaque.
```shell!
python store_each_plq.py ...
```
## 4. Merge the duplicate plaque label
```shell
python combine_multi_plque.py ... 
```

## PS
- The dcm2niix program just download from [dcm2niix](https://github.com/rordenlab/dcm2niix/releases)
