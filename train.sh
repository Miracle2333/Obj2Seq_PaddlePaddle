export FLAGS_allocator_strategy=auto_growth
job_name=obj2seq

config=configs/obj2seq/obj2seq_r50_1x_coco.yml


log_dir=log_dir/obj2seq_decay
weights=http://10.21.226.169:8088/model_final.pdparams  

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
#nohup python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval >/dev/null 2>&1 &


# 1. eval
CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}