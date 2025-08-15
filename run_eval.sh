#!/bin/bash

# =============================================================================
# 模型评估脚本 - 用于训练完成后的评价指标测试
# 支持信息抽取(ie)和问答(qa)两种任务类型
# =============================================================================

# 配置参数
TASK_TYPE="ie"        # ie | qa
MODE="glm"            # glm | glm2 | glm3
TRAIN_TYPE="lora"     # freeze | lora | ptuning | ptuning_v2
BASE_MODEL_PATH="/root/autodl-fs/chatglm6b"  # LoRA 评估时用于合并的基础模型，可用 BASE_MODEL_PATH 覆盖环境变量
if [ -n "$BASE_MODEL_PATH_ENV" ]; then
    BASE_MODEL_PATH="$BASE_MODEL_PATH_ENV"
fi

# LoRA 参数（仅在 TRAIN_TYPE=lora 时用于拼接目录与检查）
LORA_DIM=16
LORA_ALPHA=64

# 根据任务类型设置数据路径
if [ "$TASK_TYPE" = "ie" ]; then
        TEST_PATH="data/spo_1.json"
        echo "=== 信息抽取任务评估 (eval=spo_1) ==="
elif [ "$TASK_TYPE" = "qa" ]; then
        TEST_PATH="data/d2q_1.json"
        echo "=== 问答任务评估 (eval=d2q_1) ==="
else
    echo "错误: TASK_TYPE 必须是 'ie' 或 'qa'"
    exit 1
fi

# 训练方式标签与输出主目录匹配逻辑
case "$TRAIN_TYPE" in
    freeze)
        TRAIN_TYPE_TAG="Freeze";;
    lora)
        TRAIN_TYPE_TAG="Lora-d${LORA_DIM}-a${LORA_ALPHA}";;
    ptuning)
        # 原始 P-Tuning：仅 Embedding 前添加前缀
        TRAIN_TYPE_TAG="P-Tuning";;
    ptuning_v2)
        # P-Tuning-V2：层级前缀 (prefix_projection=True)
        TRAIN_TYPE_TAG="P-Tuning-V2";;
    *) echo "错误: 不支持的 TRAIN_TYPE=$TRAIN_TYPE (期望 freeze|lora|ptuning|ptuning_v2)"; exit 1;;
esac

MODEL_BASE_DIR="./output/${MODE}-${TRAIN_TYPE_TAG}-${TASK_TYPE}"
echo "模型基目录推断为: $MODEL_BASE_DIR"

# 检查测试数据是否存在
if [ ! -f "$TEST_PATH" ]; then
    echo "错误: 测试数据文件不存在: $TEST_PATH"
    exit 1
fi

# 检查模型目录是否存在
if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo "错误: 模型目录不存在: $MODEL_BASE_DIR"
    exit 1
fi

# 评估参数配置
# 如果不指定 FORCE_EVAL_SAMPLES，则自动用验证集全部样本
DATASET_SIZE=$(grep -c '' "$TEST_PATH")
if [ -n "$FORCE_EVAL_SAMPLES" ]; then
    EVAL_SAMPLES=$FORCE_EVAL_SAMPLES
else
    EVAL_SAMPLES=$DATASET_SIZE
fi
MAX_LENGTH=1024          # 最大生成长度
MAX_NEW_TOKENS=256       # 新生成 token 上限 (与 evaluate_model.py 对齐)
DEVICE="0"               # GPU设备号
DO_SAMPLE=false          # 是否采样生成 (false=贪心解码, true=采样生成)
TOP_P=0.8                # 采样参数
TEMPERATURE=0.8          # 采样温度

## 已废弃的 LoRA 合并控制参数已移除：evaluate_model.py 会在检测到“纯 adapter(只有 adapter_config.json 无 config.json)”时自动合并并写入同级目录，保留 adapter_model.bin。

# 创建评估结果目录
EVAL_RESULTS_DIR="${MODEL_BASE_DIR}/eval_results"
mkdir -p "$EVAL_RESULTS_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "================================"
echo "开始模型评估..."
echo "任务类型: $TASK_TYPE"
echo "测试数据: $TEST_PATH" 
echo "模型目录: $MODEL_BASE_DIR"
echo "评估样本数: $EVAL_SAMPLES (dataset lines=$DATASET_SIZE)"
echo "设备: GPU-$DEVICE"
echo "时间戳: $TIMESTAMP"
echo "================================"

# 查找所有训练好的模型检查点
MODEL_DIRS=$(find "$MODEL_BASE_DIR" -type d -name "*epoch*step*" | sort)

if [ -z "$MODEL_DIRS" ]; then
    echo "错误: 在 $MODEL_BASE_DIR 中未找到任何模型检查点"
    exit 1
fi

echo "找到以下模型检查点:"
echo "$MODEL_DIRS"
echo "================================"

# 创建汇总结果文件
SUMMARY_FILE="${EVAL_RESULTS_DIR}/eval_summary_${TASK_TYPE}_${TIMESTAMP}.csv"

# 写入CSV表头
if [ "$TASK_TYPE" = "ie" ]; then
    echo "model_path,samples,precision,recall,f1,bleu2,rouge1,rouge2" > "$SUMMARY_FILE"
else
    echo "model_path,samples,bleu2,rouge1,rouge2" > "$SUMMARY_FILE"
fi

# 对每个模型检查点进行评估
for MODEL_PATH in $MODEL_DIRS; do
    echo "正在评估模型: $MODEL_PATH"

    MODEL_NAME=$(basename "$MODEL_PATH")
    PRED_FILE="${EVAL_RESULTS_DIR}/pred_${MODEL_NAME}_${TIMESTAMP}.jsonl"

    # 使用数组构建命令以避免意外的反斜杠参数
    CMD=(python evaluate_model.py
        --model_path "$MODEL_PATH"
        --test_path "$TEST_PATH"
        --mode "$MODE"
        --task_type "$TASK_TYPE"
        --device "$DEVICE"
        --max_length "$MAX_LENGTH"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --eval_samples "$EVAL_SAMPLES"
        --save_predictions "$PRED_FILE"
    )

    # LoRA 适配器检测：仅当目录是“纯 adapter”才需要提供 base_model_path 以便脚本自动合并
    if [ -f "$MODEL_PATH/adapter_config.json" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
        CMD+=(--base_model_path "$BASE_MODEL_PATH")
    fi

    if [ "$DO_SAMPLE" = true ]; then
        CMD+=(--do_sample --top_p "$TOP_P" --temperature "$TEMPERATURE")
    fi

    # 展示命令（格式化输出）
    printf "执行命令: %s\n" "${CMD[*]}"

    # 运行并捕获输出
    EVAL_OUTPUT="$("${CMD[@]}" 2>&1)"
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "✓ 评估完成: $MODEL_PATH"
        echo "预测结果保存至: $PRED_FILE"
        
        # 从输出中提取指标并保存到CSV
        if [ "$TASK_TYPE" = "ie" ]; then
            # 提取 precision, recall, f1, bleu2, rouge1, rouge2
            PRECISION=$(echo "$EVAL_OUTPUT" | grep -o "precision: [0-9.]*" | cut -d' ' -f2)
            RECALL=$(echo "$EVAL_OUTPUT" | grep -o "recall: [0-9.]*" | cut -d' ' -f2)
            F1=$(echo "$EVAL_OUTPUT" | grep -o "f1: [0-9.]*" | cut -d' ' -f2)
            BLEU2=$(echo "$EVAL_OUTPUT" | grep -o "bleu2: [0-9.]*" | cut -d' ' -f2)
            ROUGE1=$(echo "$EVAL_OUTPUT" | grep -o "rouge1: [0-9.]*" | cut -d' ' -f2)
            ROUGE2=$(echo "$EVAL_OUTPUT" | grep -o "rouge2: [0-9.]*" | cut -d' ' -f2)
            SAMPLES=$(echo "$EVAL_OUTPUT" | grep -o "Samples evaluated: [0-9]*" | cut -d' ' -f3)
            
            echo "$MODEL_PATH,$SAMPLES,$PRECISION,$RECALL,$F1,$BLEU2,$ROUGE1,$ROUGE2" >> "$SUMMARY_FILE"
            
            echo "  - Precision: $PRECISION"
            echo "  - Recall: $RECALL" 
            echo "  - F1: $F1"
            echo "  - BLEU-2: $BLEU2"
            echo "  - ROUGE-1: $ROUGE1"
            echo "  - ROUGE-2: $ROUGE2"
        else
            # QA任务只有BLEU和ROUGE指标
            BLEU2=$(echo "$EVAL_OUTPUT" | grep -o "bleu2: [0-9.]*" | cut -d' ' -f2)
            ROUGE1=$(echo "$EVAL_OUTPUT" | grep -o "rouge1: [0-9.]*" | cut -d' ' -f2)
            ROUGE2=$(echo "$EVAL_OUTPUT" | grep -o "rouge2: [0-9.]*" | cut -d' ' -f2)
            SAMPLES=$(echo "$EVAL_OUTPUT" | grep -o "Samples evaluated: [0-9]*" | cut -d' ' -f3)
            
            echo "$MODEL_PATH,$SAMPLES,$BLEU2,$ROUGE1,$ROUGE2" >> "$SUMMARY_FILE"
            
            echo "  - BLEU-2: $BLEU2"
            echo "  - ROUGE-1: $ROUGE1"
            echo "  - ROUGE-2: $ROUGE2"
        fi
        
    else
        echo "✗ 评估失败: $MODEL_PATH"
    echo "错误信息:"
    echo "$EVAL_OUTPUT"
    fi
    
    echo "--------------------------------"
done

echo "================================"
echo "所有模型评估完成!"
echo "汇总结果保存至: $SUMMARY_FILE"
echo "个别模型预测结果保存在: $EVAL_RESULTS_DIR"

# 显示最终汇总结果
echo ""
echo "=== 评估结果汇总 ==="
if [ -f "$SUMMARY_FILE" ]; then
    cat "$SUMMARY_FILE"
    echo ""
    
    # 找出最佳模型（根据F1分数或BLEU分数）
    if [ "$TASK_TYPE" = "ie" ]; then
        echo "=== 最佳模型 (按F1分数) ==="
        # 跳过表头，按F1分数排序，取最高的
        tail -n +2 "$SUMMARY_FILE" | sort -t',' -k5 -nr | head -1 | while IFS=',' read -r model samples precision recall f1 bleu2 rouge1 rouge2; do
            echo "模型路径: $model"
            echo "F1分数: $f1"
            echo "Precision: $precision, Recall: $recall"
            echo "BLEU-2: $bleu2, ROUGE-1: $rouge1, ROUGE-2: $rouge2"
        done
    else
        echo "=== 最佳模型 (按BLEU-2分数) ==="
        # 跳过表头，按BLEU2分数排序，取最高的
        tail -n +2 "$SUMMARY_FILE" | sort -t',' -k3 -nr | head -1 | while IFS=',' read -r model samples bleu2 rouge1 rouge2; do
            echo "模型路径: $model"
            echo "BLEU-2: $bleu2, ROUGE-1: $rouge1, ROUGE-2: $rouge2"
        done
    fi
fi

echo "================================"
echo "评估脚本执行完成!"
echo ""
echo "快速使用提示:"
echo "1. 查看汇总结果: cat $SUMMARY_FILE"
echo "2. 查看预测示例: head -5 ${EVAL_RESULTS_DIR}/pred_*_${TIMESTAMP}.jsonl"
echo "3. 修改评估参数: 编辑脚本开头的配置部分"
echo "4. 限制生成长度: 调整 MAX_NEW_TOKENS 防止提示被截断"
