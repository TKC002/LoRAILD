tasks=(cola mrpc qnli rte)
curriculum0=normal
curriculum1s=(sharp smooth)

ild_starts=(5 6 7 8 9 10)
declare -A lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00005
        ["qnli"]=0.00002
        ["rte"]=0.00005
        ["sst2"]=0.00005
        ["stsb"]=0.00005
)
declare -A curriculum_lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.0001
        ["qnli"]=0.00002
        ["rte"]=0.0001
        ["sst2"]=0.00005
        ["stsb"]=0.00005
)

ex_num=5
filepath=train.py
all_conf=confs/all_Merged.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/method/roberta/main/CurriculumRAILKD/8.yaml
nep_token_conf=confs/nep_token.yaml

for task in ${tasks[@]}
do
    for ild_start in ${ild_starts[@]}
    do
        for curriculum1 in ${curriculum1s[@]}
        do
            RANDOM=0
            for i in `seq 1 $ex_num`
            do
                outpath="${curriculum1}/${curriculum0}/ild_start_${ild_start}/"
                accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM --ild_start $ild_start --curriculum $curriculum0 $curriculum1 --curriculum_lr ${curriculum_lrs[${task}]} --outpath $outpath
            done
        done
    done
done