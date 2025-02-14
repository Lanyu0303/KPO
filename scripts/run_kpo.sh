work_dir=''
cd $work_dir
MODEL_PATH=""
CHECKPOINT_PATH=""
DATA_PATH=""
LOG_DIR=""
OUTPUT_DIR=""
wandb_proj_name=""
wandb_run_name=""

CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --nproc_per_node 4 --master_port 2025215 ./pipeline/lora_KPO_CL.py \
            --model_name $MODEL_PATH  \
            --resume_from_checkpoint $CHECKPOINT_PATH  \
            --data_path $DATA_PATH \
            --batch_size 4 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-5 \
            --eval_step 60 \
            --beta 1.0 \
            --neg_num 9 \
            --num_train_epochs 3 \
            --logging_dir $LOG_DIR \
            --output_dir $OUTPUT_DIR \
            --wandb_project $wandb_proj_name \
            --wandb_name $wandb_run_name

