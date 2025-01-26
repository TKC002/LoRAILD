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
        ["mrpc"]=0.00005
        ["qnli"]=0.00002
        ["rte"]=0.00002
        ["sst2"]=0.00002
        ["stsb"]=0.0001
)
declare -A curriculum0s=(
        ["cola"]=normal
        ["mrpc"]=normal
        ["qnli"]=normal
        ["rte"]=normal
        ["sst2"]=normal
        ["stsb"]=normal
)
declare -A curriculum1s=(
        ["cola"]=smooth
        ["mrpc"]=smooth
        ["qnli"]=smooth
        ["rte"]=smooth
        ["sst2"]=sharp
        ["stsb"]=smooth
)
declare -A ild_starts=(
        ["cola"]=8
        ["mrpc"]=6
        ["qnli"]=7
        ["rte"]=7
        ["sst2"]=2
        ["stsb"]=4
)
ex_num=5
filepath=analyze.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/CurriculumFullLoRAILD_new2/8/random.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    for i in `seq 1 $ex_num`
    do
        python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM --ild_start ${ild_starts[${task}]} --curriculum ${curriculum0s[${task}]} ${curriculum1s[${task}]} --curriculum_lr ${curriculum_lrs[${task}]}
    done
done