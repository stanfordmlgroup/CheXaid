# 0
Model[1]='new_gt_1549918027265_4CF892/iter_6656_pulm-valid_pulm_tbAUROC_0.79.pth.tar'
# 4
Model[5]='new_gt_1549925844213_35D1C6/iter_4608_pulm-valid_pulm_tbAUROC_0.81.pth.tar'
# 2
Model[3]='new_gt_1549925844216_9135F8/iter_3072_pulm-valid_pulm_tbAUROC_0.81.pth.tar'
# 1
Model[2]='new_gt_1549928107265_48DA80/iter_4096_pulm-valid_pulm_tbAUROC_0.76.pth.tar'
# 3
Model[4]='new_gt_1549928115318_D7AA75/iter_7168_pulm-valid_pulm_tbAUROC_0.79.pth.tar'

save_folder=`date +%s`
echo $save_folder
mkdir results/$save_folder
echo ${Model[*]} > results/$save_folder/models.txt

for split in 'valid' 'test'
do
    for index in 1 2 3 4 5
    do
        python test.py --eval_pulm=True --transform_classifier=False --save_cams=False --task_sequence=pulm --ckpt_path=ckpts/${Model[index]}  --split=$split
        cp results/debugging/$split/probabilities.csv results/$save_folder/probabilities_${split}${index}.csv
    done
done
