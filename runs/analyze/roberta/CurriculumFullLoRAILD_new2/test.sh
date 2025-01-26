declare -A lrs=(
        ["cola"]=0.00001
        ["mrpc"]=0.00002
        ["qnli"]=0.00003
        ["rte"]=0.00004
        ["sst2"]=0.00005
        ["stsb"]=0.00006
)
declare -A curriculum_lrs=(
        ["cola"]=0.00001
        ["mrpc"]=0.00002
        ["qnli"]=0.00003
        ["rte"]=0.00004
        ["sst2"]=0.00005
        ["stsb"]=0.00006
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
        ["mrpc"]=smooth
        ["qnli"]=sharp
        ["rte"]=smooth
        ["sst2"]=sharp
        ["stsb"]=smooth
)
declare -A ild_starts=(
        ["cola"]=1
        ["mrpc"]=2
        ["qnli"]=3
        ["rte"]=4
        ["sst2"]=5
        ["stsb"]=6
)
ex_num=5
# filepath=train.py
# all_conf=confs/all_LoRA.yaml
# task_conf=confs/dummy_task.yaml
# method_conf=confs/method/roberta/main/CurriculumFullLoRAILD_new2//.yaml
# nep_token_conf=confs/nep_token.yaml

for task in "${!lrs[@]}"; 
do
#     RANDOM=0
    for i in `seq 1 $ex_num`
    do
        echo start
        echo ${lrs[${task}]}
        echo ${ild_starts[${task}]} 
        echo ${curriculum0s[${task}]} 
        echo ${curriculum1s[${task}]}
        echo ${curriculum_lrs[${task}]}
        echo end
    done
done