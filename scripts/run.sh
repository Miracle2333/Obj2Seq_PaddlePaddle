CONFIG_FILE='configs/detection_r50_seqhead_plus_box_refine.yaml'
OUTPUT_DIR='output_debug/'
NGPUS=8
PORT=34222

#nohup 
python3.7 -m paddle.distributed.launch --gpus 0,1,2,3 main.py \
 --cfg $CONFIG_FILE --output_dir $OUTPUT_DIR 
 #1>logs/10.log 2>logs/10.err &
