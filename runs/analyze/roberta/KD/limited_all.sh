declare -A lrs=(
        ["cola"]=0.00002
        ["mrpc"]=0.00001
        ["qnli"]=0.00002
        ["rte"]=0.00002
        ["sst2"]=0.00001
        ["stsb"]=0.00005
)
declare -A is=(
        ["cola"]=5
        ["mrpc"]=4
        ["qnli"]=2
        ["rte"]=2
        ["sst2"]=4
        ["stsb"]=4
)
filepath=analyze.py
all_conf=confs/all.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/KD.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i ${is[${task}]} --seed $RANDOM
done