tasks=(cola mrpc qnli rte sst2 stsb)
rs=(8 32 64 128)
filepath=teacher_merge.py
all_conf=confs/all_LoRA.yaml
task_conf=confs/dummy_task.yaml

nep_token_conf=confs/nep_token.yaml


for task in ${tasks[@]}
do
    for r in ${rs[@]}
    do
        method_conf=confs/teacher_merge$r.yaml
        # cat $method_conf
        # python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --task $task --workdir /gs/fs/tga-cl/suzuki.t.dp/GLUE/ --lr 0 --i 0 --seed 0 
        # python $filepath --all_conf $all_conf --task_conf $task_conf --method_conf $method_conf --nep_token $nep_token_conf --task $task --lr 0 --i 0 --seed 0 
    done
done