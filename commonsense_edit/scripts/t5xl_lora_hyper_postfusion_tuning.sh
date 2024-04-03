#Name=finetuning-m-cb+cf+a
#
#print_log=/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/log/${Name}
#if [ -d $print_log ]; then
#    # 如果folder文件夹已经存在，则先删除它
#    rm -r $print_log
#fi
#mkdir -p $print_log

# 预训练
# python /media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/train.py experiment=commonsense model=t5large_pretrain editor=ft \
#      | tee ${print_log}/print_record.txt

#
python /media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/train_t5-xl.py experiment=commonsense model=t5xl_train_lora_hyperdecoders_postfusion editor=lora_hyperdecoders_postfusion \
#      | tee ${print_log}/print_record.txt