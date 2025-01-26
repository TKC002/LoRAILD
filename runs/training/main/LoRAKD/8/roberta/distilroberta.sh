declare -A lrs=(
        ["cola"]=0.00005
        ["mrpc"]=0.00005
        ["qnli"]=0.00002
        ["rte"]=0.00005
        ["sst2"]=0.00005
        ["stsb"]=0.00005
)


ex_num=5
filepath=train.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/method/roberta/main/LoRAKD/8/roberta/distilroberta.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    for i in `seq 1 $ex_num`
    do
        accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM
    done
done