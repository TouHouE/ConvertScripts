pip install openpyxl

#export TXT_MNT="$SHARE/usbB/HsinChu/text:/workspace/text"
#export CT_MNT="$SHARE/data/image/HsinChu:/workspace/ct"
#export OUT_MNT="$SHARE/data:/workspace/data"
#export PROJ_MNT="$HSU/workspace/project:/workspace/project"


python /workspace/project/ConvertScripts/prompt_generate.py \
--report_file=/workspace/text/CT_Report.xlsx \
 --ct_root=/workspace/ct/batch2-13,/workspace/ct/batch2-14,/workspace/ct/batch2-16,/workspace/ct/batch2-26,/workspace/ct/batch2-232 \
--prompt_template_path=/workspace/project/ConvertScripts/template.json \
--json_path=/workspace/data/conversations_b2_2.json
