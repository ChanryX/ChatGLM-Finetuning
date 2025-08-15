#!/bin/bash

# 任务类型开关: ie (信息抽取) 或 qa (问答)
TASK_TYPE="qa"  # ie | qa

# 训练方式开关: freeze | lora | ptuning | ptuning_v2
TRAIN_TYPE="lora"

# LoRA 参数（仅在 TRAIN_TYPE=lora 时使用）
LORA_DIM=16
LORA_ALPHA=64
LORA_DROPOUT=0.1
LORA_MODULE_NAME="query_key_value"

# 根据任务类型设置不同的参数
if [ "$TASK_TYPE" = "ie" ]; then
        TRAIN_PATH="data/spo_0.json"
        EVAL_PATH="data/spo_1.json"
        echo "Running Information Extraction task (train=spo_0, eval=spo_1)..."
elif [ "$TASK_TYPE" = "qa" ]; then
        TRAIN_PATH="data/d2q_0.json"
        EVAL_PATH="data/d2q_1.json"
        echo "Running Question Answering task (train=d2q_0, eval=d2q_1)..."
else
    echo "Error: TASK_TYPE must be 'ie' or 'qa'"
    exit 1
fi

# 训练方式标签映射（用于输出目录命名）
case "$TRAIN_TYPE" in
    freeze)
        TRAIN_TYPE_TAG="Freeze";;
    lora)
        TRAIN_TYPE_TAG="Lora-d${LORA_DIM}-a${LORA_ALPHA}";;
    ptuning)
        # 原始 P-Tuning：仅在 Embedding 前注入前缀
        TRAIN_TYPE_TAG="P-Tuning";;
    ptuning_v2)
        # P-Tuning-V2：prefix_projection=True，层级前缀（更大参数容量）
        TRAIN_TYPE_TAG="P-Tuning-V2";;
    *)
        echo "Error: unsupported TRAIN_TYPE=$TRAIN_TYPE (expect freeze|lora|ptuning|ptuning_v2)"; exit 1;;
esac

# 输出目录：output/<model>-<traintype>-<task>
MODEL_TAG="glm"   # 如需切换 glm2 / glm3 可改这里
OUTPUT_DIR="./output/${MODEL_TAG}-${TRAIN_TYPE_TAG}-${TASK_TYPE}"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 自动适配验证集行数作为默认评估样本数（可通过环境变量 FORCE_EVAL_SAMPLES 覆盖）
EVAL_SET_SIZE=$(grep -c '' "$EVAL_PATH")
if [ -n "$FORCE_EVAL_SAMPLES" ]; then
    EFFECTIVE_EVAL_SAMPLES=$FORCE_EVAL_SAMPLES
else
    EFFECTIVE_EVAL_SAMPLES=$EVAL_SET_SIZE
fi
echo "Eval set lines: $EVAL_SET_SIZE, using eval_samples=$EFFECTIVE_EVAL_SAMPLES"

# 评估参数提示：
# --eval_on_train            训练集随机采样评估
# --eval_print_samples 3     打印若干评估样本
# --gen_max_length 256       限制生成长度加速评估
# --eval_log_file            评估日志输出文件

# GPU 设置：可通过环境变量 GPU_IDS 覆盖 (例如 GPU_IDS=0,1,2,3 bash run_freeze.sh)
GPU_IDS=${GPU_IDS:-0,1}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

CMD=(
    deepspeed --master_port 520 train.py
    --train_path "$TRAIN_PATH"
    --eval_path "$EVAL_PATH"
    --task_type "$TASK_TYPE"
    --eval_step 20
    --eval_samples "$EFFECTIVE_EVAL_SAMPLES"
    --eval_on_train
    --eval_print_samples 3
    --max_new_tokens 256
    --eval_log_file "${OUTPUT_DIR}/eval.log"
    --model_name_or_path /root/autodl-fs/chatglm6b/
    --per_device_train_batch_size 1
    --max_len 1560
    --max_src_len 1024
    --learning_rate 1e-4
    --weight_decay 0.1
    --num_train_epochs 2
    --gradient_accumulation_steps 4
    --warmup_ratio 0.1
    --mode glm
    --train_type "$TRAIN_TYPE"
    --seed 1234
    --ds_file ds_zero2_no_offload.json
    --gradient_checkpointing
    --show_loss_step 10
    --output_dir "$OUTPUT_DIR"
    --tb_log_dir "runs/$(basename "$OUTPUT_DIR")"
) # tensorboard --logdir runs

# 按训练方式追加特定参数
case "$TRAIN_TYPE" in
    freeze)
        CMD+=(--freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24.")
        ;;
    lora)
        CMD+=(--lora_dim $LORA_DIM --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --lora_module_name $LORA_MODULE_NAME)
        ;;
    ptuning)
        CMD+=(--pre_seq_len 16 --prefix_projection False)
        ;;
    ptuning_v2)
        CMD+=(--pre_seq_len 16 --prefix_projection True)
        ;;
esac

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Running command:"
printf '%q ' "${CMD[@]}"; echo

"${CMD[@]}"