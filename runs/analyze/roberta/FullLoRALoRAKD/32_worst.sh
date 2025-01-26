declare -A lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00005
        ["qnli"]=0.00002
        ["rte"]=0.00002
        ["sst2"]=0.00005
        ["stsb"]=0.0001
)
declare -A is=(
        ["cola"]=1
        ["mrpc"]=5
        ["qnli"]=3
        ["rte"]=1
        ["sst2"]=3
        ["stsb"]=4
)


ex_num=5
filepath=analyze.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/analyze/roberta/FullLoRALoRAKD/32.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i ${is[${task}]} --seed $RANDOM --outpath 'worst'
done