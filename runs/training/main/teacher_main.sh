declare -A lrs=(
        ["cola"]=0.00001
        ["mrpc"]=0.00002
        ["qnli"]=0.00001
        ["rte"]=0.00001
        ["sst2"]=0.00001
        ["stsb"]=0.00002
)


ex_num=5
filepath=$1
all_conf=$2
task_conf=$3
method_conf=$4
nep_token_conf=$5

for task in "${!lrs[@]}"; 
do
    RANDOM=0
    for i in `seq 1 $ex_num`
    do
        accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr ${lrs[${task}]} --task $task --i $i --seed $RANDOM
    done
done