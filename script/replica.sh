
cd ..
echo "当前工作目录: $(pwd)"
dataset=replica_900
path=./data/${dataset}
scene='room_0' 

output=./output/${dataset}/${scene} 
# output=/media/shy/data/decom-2dgs/server-replica-900/contra_wo_bg_sample/${dataset}/${scene} 
source=${path}/${scene}
contra_iter=20000

opt=$1

if test $opt == "train_rgb"; then
    python train_gaus.py \
        -s ${source} \
        -m ${output} 

elif test $opt == "merge_patches"; then
    python merge_patches.py \
        -s ${source}  \
        -m ${output} 

elif test $opt == "remove_ab_gaus"; then
    python remove_ab_gaus \
        -s ${source}\
        -m ${output} \
        --sam_folder split \
        --iterations 9000 \
        --split \
        --prune  \
        --split_cycle_interval 1200

elif test $opt == "train_contra"; then
    python train.py \
        -s ${source} \
        -m ${output} \
        --iterations $contra_iter\
        --start_checkpoint ${output}/chkpntsplit.pth\
        --sam_folder split \
        --include_feature \
        --save_name sp_

elif test $opt == "eval"; then
    python evaluation/eval_nvs.py \
        -s ${source} \
        -m ${output} \
        --save_path ./output/test/split/\
        --start_checkpoint ${output}/split/chkpnt/sp_${contra_iter}.pth 
        
elif test $opt == "eval_3d"; then
    python evaluation/eval_3d.py \
        -s ${source} \
        -m ${output} \
        --save_path ./output/test_3d/split/\
        --result_save_path ./output/test_3d\
        --clean_history \
        --method split \
        --start_checkpoint ${output}/split/chkpnt/sp_${contra_iter}.pth \

fi
