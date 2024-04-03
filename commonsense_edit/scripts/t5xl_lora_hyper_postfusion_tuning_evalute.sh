#Name=finetuning-m-cb+cf+a
#
#print_log=/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/log/${Name}
#if [ -d $print_log ]; then
#    # 如果folder文件夹已经存在，则先删除它
#    rm -r $print_log
#fi
#mkdir -p $print_log


python /media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/evaluate_t5-xl.py experiment=commonsense model=t5xl_train_lora_hyperdecoders_postfusion editor=lora_hyperdecoders_postfusion \
