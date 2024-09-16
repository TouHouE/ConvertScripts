# Taipei-1 Usage Convert Scripts
## 1. Using to convert *.dcm file to *.nii.gz file
- `dcm2niix_HsinChu.py`
  - This script is using to convert *.dcm file that come from HsinChu Hospital
  - To launch this script successful, you should place the `scripts_class.py` at the same path.
  - The dictionary from HsinChu be like:
    ```
    HsinChu
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
    - ex: <the path for patient>/<series number>/<instance uid>/<cardiac phase>/<name generated by dcm2niix.exe>.nii.gz
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
## 2. Generate the prompt file
- Using `prompt_generate.py` to generate the VQA prompt 
- This script should be launch after `dcm2niix_HsinChu.py` processed.
- The `--report_file` is placed at /mnt/usbB/HsinChu/text/*.xlsx
  - I do some rename, if you see "batch1" in file name, this table should match the batch1 nii folder
  - But, if is batch2, you should combine all of 3 table as 1 table then running the `prompt_generate.py`
- The `prompt_scripts_utils.py` should place at same path.
## PS
1. The dcm2niix program just download from [dcm2niix](https://github.com/rordenlab/dcm2niix/releases)