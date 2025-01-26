declare -A lrs=(
        ["cola"]=0.0000
        ["mrpc"]=0.0000
        ["qnli"]=0.0000
        ["rte"]=0.0000
        ["sst2"]=0.0000
        ["stsb"]=0.0000
)
declare -A curriculum_lrs=(
        ["cola"]=0.0000
        ["mrpc"]=0.0000
        ["qnli"]=0.0000
        ["rte"]=0.0000
        ["sst2"]=0.0000
        ["stsb"]=0.0000
)
declare -A curriculum0s=(
        ["cola"]=normal
        ["mrpc"]=normal
        ["qnli"]=normal
        ["rte"]=
        ["sst2"]=normal
        ["stsb"]=normal
)
declare -A curriculum1s=(
        ["cola"]=
        ["mrpc"]=
        ["qnli"]=
        ["rte"]=
        ["sst2"]=
        ["stsb"]=
)
declare -A ild_starts=(
        ["cola"]=
        ["mrpc"]=
        ["qnli"]=
        ["rte"]=
        ["sst2"]=
        ["stsb"]=
)
declare -A is=(
        ["cola"]=
        ["mrpc"]=
        ["qnli"]=
        ["rte"]=
        ["sst2"]=
        ["stsb"]=
)
ex_num=5
filepath=train.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml
method_conf=confs/method/roberta/main/CurriculumFullLoRAILD_new2//.yaml
nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    for i in `seq 1 $ex_num`
    do
        accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM --ild_start ${ild_starts[${task}]} --curriculum ${curriculum0s[${task}]} ${curriculum1s[${task}]} --curriculum_lr ${curriculum_lrs[${task}]}
    done
done