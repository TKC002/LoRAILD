declare -A lrs=(
        ["cola"]=0.00000
        ["mrpc"]=0.00000
        ["qnli"]=0.00000
        ["rte"]=0.00000
        ["sst2"]=0.00000
        ["stsb"]=0.00000
)
declare -A is=(
        ["cola"]=99
        ["mrpc"]=99
        ["qnli"]=99
        ["rte"]=99
        ["sst2"]=99
        ["stsb"]=99
)
filepath=analyze.py
all_conf=confs/all.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/no_train/teacher.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i ${is[${task}]} --seed $RANDOM
done