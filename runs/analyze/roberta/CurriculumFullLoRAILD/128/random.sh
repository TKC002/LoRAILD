declare -A lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00005
        ["qnli"]=0.00005
        ["rte"]=0.00005
        ["sst2"]=0.0001
        ["stsb"]=0.0001
)
declare -A curriculum_lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00001
        ["qnli"]=0.00002
        ["rte"]=0.00002
        ["sst2"]=0.00005
        ["stsb"]=0.0001
)
declare -A curriculum0s=(
        ["cola"]=normal
        ["mrpc"]=normal
        ["qnli"]=normal
        ["rte"]=kd_rem
        ["sst2"]=normal
        ["stsb"]=normal
)
declare -A curriculum1s=(
        ["cola"]=sharp
        ["mrpc"]=sharp
        ["qnli"]=smooth
        ["rte"]=smooth
        ["sst2"]=smooth
        ["stsb"]=sharp
)
declare -A ild_starts=(
        ["cola"]=5
        ["mrpc"]=6
        ["qnli"]=5
        ["rte"]=9
        ["sst2"]=3
        ["stsb"]=2
)
ex_num=5
filepath=analyze.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/CurriculumFullLoRAILD/128/random.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    for i in `seq 1 $ex_num`
    do
        python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM --ild_start ${ild_starts[${task}]} --curriculum ${curriculum0s[${task}]} ${curriculum1s[${task}]} --curriculum_lr ${curriculum_lrs[${task}]}
    done
done