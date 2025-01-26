ild_method=average
linear=32
declare -A lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00005
        ["rte"]=0.00005
        ["stsb"]=0.00005
        ["qnli"]=0.00002
        ["sst2"]=0.00002
)
declare -A is=(
        ["cola"]=2
        ["mrpc"]=2
        ["qnli"]=4
        ["rte"]=5
        ["sst2"]=3
        ["stsb"]=4
)
filepath=analyze.py
all_conf=confs/all_Merged.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/RAILKD_new/$linear.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i ${is[${task}]} --seed $RANDOM --outpath worst
done