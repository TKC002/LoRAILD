task=rte
curriculum0=kd_rem
curriculum1s=(sharp smooth)

ild_starts=(5 6 7 8 9 10)
lr=0.00005
curriculum_lr=0.0001

ex_num=5
filepath=train.py
all_conf=confs/all_Merged.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/method/roberta/main/CurriculumRAILKD/8.yaml
nep_token_conf=confs/nep_token.yaml

for ild_start in ${ild_starts[@]}
do
    for curriculum1 in ${curriculum1s[@]}
    do
        RANDOM=0
        for i in `seq 1 $ex_num`
        do
            outpath="${curriculum1}/${curriculum0}/ild_start_${ild_start}/"
            accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr $lr --task $task --i $i --seed $RANDOM --ild_start $ild_start --curriculum $curriculum0 $curriculum1 --curriculum_lr $curriculum_lr --outpath $outpath
        done
    done

done