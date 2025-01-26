tasks=(cola mrpc qnli rte sst2 stsb)
lrs=(0.00001 0.00002 0.00005 0.0001 0.0002 0.0005)
ex_num=5
filepath=$1
all_conf=$2
task_conf=$3
method_conf=$4
nep_token_conf=$5
for task in ${tasks[@]}
do
    for lr in ${lrs[@]}
    do
        RANDOM=0
        for i in `seq 1 $ex_num`
        do
            accelerate launch $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --lr $lr --task $task --i $i --seed $RANDOM
        done
    done
done