
job_name=obj2seq

config=configs/obj2seq/obj2seq_r50_1x_coco.yml


log_dir=log_dir/obj2seq_debug
weights=output/distill_tea_ppdino_swin_to_r50vd_12e/0.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
#nohup
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} 
#>/dev/null 2>&1 &
