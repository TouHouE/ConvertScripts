pip install openpyxl

python /workspace/project/ConvertScripts/prompt_generate.py \
--report_file=/workspace/text/ccta_report_batch_1.xlsx --ct_root=/workspace/ct/batch1 \
--prompt_template_path=/workspace/project/ConvertScripts/template.json \
--json_path=/workspace/data/conversations_b1.json