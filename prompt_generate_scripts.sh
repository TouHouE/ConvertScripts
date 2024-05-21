pip install openpyxl

python /workspace/project/ConvertScripts/prompt_generate.py \
--report_file=/workspace/text/ccta_report_batch_2_part1.xlsx,/workspace/text/ccta_report_batch_2_part2.xlsx,/workspace/text/ccta_report_batch_2_part3.xlsx \
 --ct_root=/workspace/ct/batch2-13,/workspace/ct/batch2-14,/workspace/ct/batch2-16,/workspace/ct/batch2-26,/workspace/ctbatch2-232 \
--prompt_template_path=/workspace/project/ConvertScripts/template.json \
--json_path=/workspace/data/conversations_b1.json