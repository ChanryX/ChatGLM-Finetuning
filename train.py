# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/8/6 16:13
"""
    文件说明：
            
"""
import argparse
import json
import math
import csv
import os
import datetime
import random  # for random sampling in evaluation
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model
from utils import DataCollator
from peft import LoraConfig, get_peft_model
from model import MODE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def normalize_text(text: str) -> str:
    """标准化文本"""
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("\r", "").replace("\t", "").strip()
    return s


def extract_triplet_lines(text: str) -> list:
    """从模型输出中抽取三元组行：包含 '_' 的行"""
    if not text:
        return []
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and "_" in ln]
    return lines


def compute_ie_counts(pred_lines: list, ref_lines: list) -> tuple:
    """返回 (TP, FP, FN) 基于行级精确匹配"""
    pred_set = set(pred_lines)
    ref_set = set(ref_lines)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    return tp, fp, fn


def _get_char_ngrams(s: str, n: int) -> Counter:
    """获取字符级n-gram"""
    s = normalize_text(s)
    s = s.replace(" ", "").replace("\n", "")
    grams = [s[i : i + n] for i in range(0, max(len(s) - n + 1, 0))]
    return Counter(grams)


def compute_corpus_bleu_cumulative(pred_texts: list, ref_texts: list, max_n: int) -> float:
    """计算累积 BLEU-N 字符级"""
    if not pred_texts or not ref_texts:
        return 0.0
    
    total_pred_len = 0
    total_ref_len = 0
    p_ns = []
    
    for n in range(1, max_n + 1):
        overlap = 0
        pred_total = 0
        for pred, ref in zip(pred_texts, ref_texts):
            pred_ngrams = _get_char_ngrams(pred, n)
            ref_ngrams = _get_char_ngrams(ref, n)
            pred_total += sum(pred_ngrams.values())
            for g, c in pred_ngrams.items():
                overlap += min(c, ref_ngrams.get(g, 0))
        p_n = (overlap / pred_total) if pred_total > 0 else 0.0
        p_ns.append(p_n)

    for pred, ref in zip(pred_texts, ref_texts):
        total_pred_len += len(normalize_text(pred).replace(" ", "").replace("\n", ""))
        total_ref_len += len(normalize_text(ref).replace(" ", "").replace("\n", ""))
    
    if total_pred_len == 0:
        return 0.0
    
    if total_pred_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (total_ref_len / max(total_pred_len, 1)))

    eps = 1e-12
    score = bp * math.exp(sum(math.log(max(p, eps)) for p in p_ns) / len(p_ns))
    return score


def compute_corpus_rouge_n(pred_texts: list, ref_texts: list, n: int) -> float:
    """ROUGE-N (recall) 字符级"""
    if not pred_texts or not ref_texts:
        return 0.0
    
    overlap = 0
    ref_total = 0
    for pred, ref in zip(pred_texts, ref_texts):
        pred_ngrams = _get_char_ngrams(pred, n)
        ref_ngrams = _get_char_ngrams(ref, n)
        ref_total += sum(ref_ngrams.values())
        for g, c in ref_ngrams.items():
            overlap += min(c, pred_ngrams.get(g, 0))
    return (overlap / ref_total) if ref_total > 0 else 0.0


def evaluate_on_samples(model, tokenizer, eval_samples: list, task_type: str, args, max_eval_samples: int = None):
    """在样本集上进行评估（rank0）。支持随机子集抽样。
    返回: (metrics, examples, elapsed_seconds)
    examples: [{'instruction':..., 'input':..., 'reference':..., 'prediction':...}, ...]
    """
    if not eval_samples:
        return {}, [], 0.0
    if hasattr(model, 'module'):
        eval_model = model.module
    else:
        eval_model = model
    # 采样
    subset = eval_samples
    if max_eval_samples is not None and len(subset) > max_eval_samples:
        subset = random.sample(subset, max_eval_samples)
    eval_model.eval()
    predictions, references = [], []
    micro_tp = micro_fp = micro_fn = 0
    legacy_cap = min(getattr(args, 'gen_max_length', 512), args.max_len)
    do_sample = getattr(args, 'gen_do_sample', False)
    temperature = getattr(args, 'gen_temperature', 0.8)
    top_p = getattr(args, 'gen_top_p', 0.8)
    import time
    t0 = time.time()
    with torch.no_grad():
        for i, sample in enumerate(subset):
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            ref = normalize_text(sample.get("output", ""))
            prompt = f"{instruction}{input_text}"
            try:
                prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(eval_model.device)
                prompt_len = prompt_ids.shape[-1]
                dynamic_cap = min(args.max_len, prompt_len + getattr(args, 'max_new_tokens', 256))
                max_length_final = min(dynamic_cap, legacy_cap) if legacy_cap else dynamic_cap
                gen_kwargs = dict(max_length=max_length_final)
                if do_sample:
                    gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
                else:
                    gen_kwargs.update(dict(do_sample=False))
                result, _ = eval_model.chat(tokenizer, prompt, **gen_kwargs)
                pred = normalize_text(result)
            except Exception as e:
                print_rank_0(f"Eval error sample {i}: {e}", args.global_rank)
                pred = ""
            if i % 5 == 0:
                print_rank_0(f"[EvalProgress] {i+1}/{len(subset)}", args.global_rank)
            predictions.append(pred)
            references.append(ref)
            if task_type == 'ie':
                pred_lines = extract_triplet_lines(pred)
                ref_lines = extract_triplet_lines(ref)
                tp, fp, fn = compute_ie_counts(pred_lines, ref_lines)
                micro_tp += tp
                micro_fp += fp
                micro_fn += fn
    elapsed = time.time() - t0
    metrics = {}
    if task_type == 'ie':
        precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.update(dict(precision=precision, recall=recall, f1=f1))
    bleu2 = compute_corpus_bleu_cumulative(predictions, references, max_n=2)
    rouge1 = compute_corpus_rouge_n(predictions, references, n=1)
    rouge2 = compute_corpus_rouge_n(predictions, references, n=2)
    metrics.update(dict(bleu2=bleu2, rouge1=rouge1, rouge2=rouge2))
    eval_model.train()
    # 组织示例（只截取前 eval_print_samples 条）
    k = getattr(args, 'eval_print_samples', 0)
    examples = []
    if k > 0:
        for i in range(min(k, len(predictions))):
            examples.append({
                'idx': i,
                'instruction': subset[i].get('instruction', ''),
                'input': subset[i].get('input', ''),
                'reference': references[i],
                'prediction': predictions[i]
            })
    return metrics, examples, elapsed


def save_metrics_to_csv(metrics: dict, csv_path: str, epoch: int, step: int):
    """将指标保存到CSV文件"""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'step'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {'epoch': epoch, 'step': step}
        row.update(metrics)
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    # DataSet
    parser.add_argument("--train_path", default="", type=str, help="")
    parser.add_argument("--eval_path", default="", type=str, help="评估数据集路径")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    # Evaluation
    parser.add_argument("--task_type", type=str, default="ie", choices=["ie", "qa"], help="任务类型：ie(信息抽取) 或 qa(问答)")
    parser.add_argument("--eval_step", type=int, default=50, help="每多少(全局)更新步进行一次评估")
    parser.add_argument("--eval_samples", type=int, default=100, help="评估样本数量上限")
    parser.add_argument("--eval_on_train", action='store_true', help="是否对训练集随机采样评估")
    parser.add_argument("--gen_max_length", type=int, default=512, help="(兼容旧参数) 旧式固定总长上限；实际使用动态长度")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="新增生成token上限: 实际max_length = min(max_len, prompt_len + max_new_tokens, gen_max_length)")
    parser.add_argument("--gen_do_sample", action='store_true', help="评估时是否采样生成")
    parser.add_argument("--gen_temperature", type=float, default=0.8, help="评估temperature")
    parser.add_argument("--gen_top_p", type=float, default=0.8, help="评估top_p")
    parser.add_argument("--eval_log_file", type=str, default=None, help="评估日志文件（默认 output_dir/eval.log）")
    parser.add_argument("--eval_print_samples", type=int, default=0, help="每次评估打印的样本数(预测+参考)")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--mode", type=str, default="glm2", help="")
    parser.add_argument("--train_type", type=str, default="lora", help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # TensorBoard 日志目录（默认: runs/<output_dir_basename> 或标准默认格式）
    parser.add_argument("--tb_log_dir", type=str, default=None, help="自定义TensorBoard日志目录；若不指定将使用 runs/<output_basename> 或框架默认")
    # deepspeed features
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # LoRA
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument("--lora_module_name", type=str, default="query_key_value", help="")
    # Freeze
    parser.add_argument("--freeze_module_name", type=str, default="layers.27.", help="")
    # P-tuning
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)

    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps

    if args.global_rank <= 0:
        # 设定 TensorBoard 日志目录：优先 --tb_log_dir；否则根据 output_dir 自动命名，避免难懂的随机时间戳目录
        if args.tb_log_dir:
            tb_dir = args.tb_log_dir
        elif args.output_dir:
            base = os.path.basename(os.path.normpath(args.output_dir))
            tb_dir = os.path.join('runs', base)
        else:
            tb_dir = None  # 使用默认模式 (runs/<timestamp>...)
        if tb_dir:
            os.makedirs(tb_dir, exist_ok=True)
            print_rank_0(f"TensorBoard log dir: {tb_dir}", args.global_rank)
            tb_write = SummaryWriter(log_dir=tb_dir)
        else:
            tb_write = SummaryWriter()
        # 创建CSV文件用于保存指标，添加时间戳避免覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"training_metrics_{timestamp}.csv"
        csv_path = os.path.join(args.output_dir if args.output_dir else ".", csv_filename)
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        print_rank_0(f"Metrics will be saved to: {csv_path}", args.global_rank)
        # 评估日志文件
        if args.eval_log_file is None:
            args.eval_log_file = os.path.join(args.output_dir if args.output_dir else ".", "eval.log")
        os.makedirs(os.path.dirname(args.eval_log_file), exist_ok=True)
        with open(args.eval_log_file, 'a', encoding='utf-8') as lf:
            lf.write(f"# Eval Log started {timestamp}\n")
            lf.write("# time\tepoch\tstep\tscope\t" \
                     "precision\trecall\tf1\tbleu2\trouge1\trouge2\telapsed_s\n")

    # 加载评估数据
    eval_samples = []
    train_raw_samples = []
    if args.eval_path and os.path.exists(args.eval_path):
        with open(args.eval_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line.strip())
                except Exception:
                    continue
                eval_samples.append(sample)
                if len(eval_samples) >= args.eval_samples:
                    break
        print_rank_0(f"Loaded {len(eval_samples)} evaluation samples", args.global_rank)
    print_rank_0(f"Eval dataset file: {args.eval_path}", args.global_rank)
    # 读取训练原始样本供随机评估
    if args.train_path and os.path.exists(args.train_path):
        with open(args.train_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    train_raw_samples.append(json.loads(line.strip()))
                except Exception:
                    continue
    print_rank_0(f"Train dataset file: {args.train_path}", args.global_rank)
    if args.global_rank <= 0:
        print_rank_0(f"Loaded {len(train_raw_samples)} training samples for potential eval", args.global_rank)
    else:
        print_rank_0("No evaluation data provided, skipping evaluation", args.global_rank)

    set_random_seed(args.seed)
    torch.distributed.barrier()
    # load tokenizer
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_name_or_path)
    print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)

    # load model
    if args.train_type == "lora":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        lora_module_name = args.lora_module_name.split(",")
        config = LoraConfig(r=args.lora_dim,
                            lora_alpha=args.lora_alpha,
                            target_modules=lora_module_name,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False,
                            )
        model = get_peft_model(model, config)
        model.config.torch_dtype = torch.float32
    elif args.train_type == "freeze":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        freeze_module_name = args.freeze_module_name.split(",")
        for name, param in model.named_parameters():
            if not any(nd in name for nd in freeze_module_name):
                param.requires_grad = False
    elif args.train_type in ["ptuning", "ptuning_v2"]:
        # P-Tuning / P-Tuning-V2 说明：
        #  - 当 prefix_projection = False (或 train_type=ptuning) 时：经典 P-Tuning，仅在 Embedding 前增加可学习前缀参数。
        #  - 当 prefix_projection = True  (或 train_type=ptuning_v2) 时：P-Tuning-V2，在每一 Transformer 层前都注入可学习前缀（更大参数容量）。
        config = MODE[args.mode]["config"].from_pretrained(args.model_name_or_path)
        config.pre_seq_len = args.pre_seq_len
        # 根据 train_type 强制对 prefix_projection 做一致性规范，若命令行传入与预期不一致则覆盖并提示。
        expected = True if args.train_type == "ptuning_v2" else False
        if args.prefix_projection != expected:
            print_rank_0(
                f"[Info] train_type={args.train_type} 期望 prefix_projection={expected} 但收到 {args.prefix_projection} ，已自动覆盖。",
                args.global_rank
            )
        config.prefix_projection = expected
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path, config=config)
        variant_name = "P-Tuning-V2" if config.prefix_projection else "P-Tuning"
        print_rank_0(f"Using {variant_name}: pre_seq_len={config.pre_seq_len}, prefix_projection={config.prefix_projection}", args.global_rank)
        # 只训练前缀相关参数（通常名称含 prefix_encoder）
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["prefix_encoder"]):
                param.requires_grad = False
    elif args.train_type == "all":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
    else:
        raise Exception("train_type无效")

    # load data
    train_dataset = MODE[args.mode]["dataset"](args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)

    # load optimizer
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1
    num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    print_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    # print parameters
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print_rank_0(name, 0)
    print_trainable_parameters(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init deepspeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config,
                                                             dist_init_required=True)
    model.train()
    tr_loss, logging_loss, min_loss = 0.0, 0.0, float('inf')
    global_step = 0
    # train
    for epoch in range(args.num_train_epochs):
        print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(epoch + 1, args.num_train_epochs,
                                                                               len(train_dataloader)), args.global_rank)
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            tr_loss += loss.item()
            model.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # write loss
                if global_step % args.show_loss_step == 0:
                    current_loss = (tr_loss - logging_loss) / (args.show_loss_step * args.gradient_accumulation_steps)
                    print_rank_0("Epoch: {}, step: {}, global_step:{}, loss: {}".format(epoch, step + 1, global_step, current_loss), args.global_rank)
                    print_rank_0("step: {}-{}-{}".format(step + 1, global_step, model.global_steps), args.global_rank)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", current_loss, global_step)
                        logging_loss = tr_loss

                # 评估指标 - 训练过程中每50步评估一次
                if eval_samples and global_step % args.eval_step == 0 and args.global_rank <= 0:
                    print_rank_0(f"[Eval] global_step {global_step}", args.global_rank)
                    metrics, examples, elapsed = evaluate_on_samples(
                        model, tokenizer, eval_samples, args.task_type, args,
                        max_eval_samples=min(args.eval_samples, len(eval_samples)))
                    if metrics:
                        per_sample_t = elapsed / max(1, min(args.eval_samples, len(eval_samples)))
                        print_rank_0(f"[Eval] elapsed {elapsed:.2f}s ({per_sample_t:.2f}s/sample)", args.global_rank)
                        for metric_name, metric_value in metrics.items():
                            tb_write.add_scalar(f"eval/{metric_name}", metric_value, global_step)
                        save_metrics_to_csv(metrics, csv_path, epoch + 1, global_step)
                        show_str = ' | '.join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                        print_rank_0(f"Eval metrics: {show_str}", args.global_rank)
                        # 写日志
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(args.eval_log_file, 'a', encoding='utf-8') as lf:
                            lf.write(f"{now}\t{epoch+1}\t{global_step}\teval\t" \
                                     f"{metrics.get('precision','')}\t{metrics.get('recall','')}\t{metrics.get('f1','')}\t" \
                                     f"{metrics.get('bleu2','')}\t{metrics.get('rouge1','')}\t{metrics.get('rouge2','')}\t{elapsed:.2f}\n")
                        # 打印样本
                        for ex in examples:
                            print_rank_0(f"[EvalSample {ex['idx']}] REF: {ex['reference']} || PRED: {ex['prediction']}", args.global_rank)

                if args.eval_on_train and train_raw_samples and global_step % args.eval_step == 0 and args.global_rank <= 0:
                    print_rank_0(f"[TrainEval] global_step {global_step}", args.global_rank)
                    train_metrics, train_examples, train_elapsed = evaluate_on_samples(
                        model, tokenizer, train_raw_samples, args.task_type, args,
                        max_eval_samples=min(args.eval_samples, len(train_raw_samples)))
                    if train_metrics:
                        per_sample_t = train_elapsed / max(1, min(args.eval_samples, len(train_raw_samples)))
                        print_rank_0(f"[TrainEval] elapsed {train_elapsed:.2f}s ({per_sample_t:.2f}s/sample)", args.global_rank)
                        for metric_name, metric_value in train_metrics.items():
                            tb_write.add_scalar(f"train_eval/{metric_name}", metric_value, global_step)
                        tagged = {f'train_{k}': v for k, v in train_metrics.items()}
                        save_metrics_to_csv(tagged, csv_path, epoch + 1, global_step)
                        show_str = ' | '.join([f"{k}:{v:.4f}" for k, v in train_metrics.items()])
                        print_rank_0(f"Train sample eval metrics: {show_str}", args.global_rank)
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(args.eval_log_file, 'a', encoding='utf-8') as lf:
                            lf.write(f"{now}\t{epoch+1}\t{global_step}\ttrain_eval\t" \
                                     f"{train_metrics.get('precision','')}\t{train_metrics.get('recall','')}\t{train_metrics.get('f1','')}\t" \
                                     f"{train_metrics.get('bleu2','')}\t{train_metrics.get('rouge1','')}\t{train_metrics.get('rouge2','')}\t{train_elapsed:.2f}\n")
                        for ex in train_examples:
                            print_rank_0(f"[TrainEvalSample {ex['idx']}] REF: {ex['reference']} || PRED: {ex['prediction']}", args.global_rank)

                # save model
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    # 若zero3训练，模型参数需要合并保存
                    if ds_config["zero_optimization"]["stage"] == 3:
                        state_dict = model._zero3_consolidated_16bit_state_dict()
                        if args.global_rank <= 0:
                            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}",
                                       state_dict)
                    else:
                        if args.global_rank <= 0:
                            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    model.train()

        # Epoch结束时的最终评估
        if eval_samples:
            print_rank_0(f"Final evaluation at end of epoch {epoch + 1}...", args.global_rank)
            metrics, examples, elapsed = evaluate_on_samples(model, tokenizer, eval_samples, args.task_type, args)
            
            # 打印指标
            print_rank_0("=" * 60, args.global_rank)
            print_rank_0(f"*** EPOCH {epoch + 1} FINAL RESULTS ***", args.global_rank)
            if args.task_type == "ie":
                print_rank_0(f"Precision: {metrics.get('precision', 0):.4f}", args.global_rank)
                print_rank_0(f"Recall: {metrics.get('recall', 0):.4f}", args.global_rank)
                print_rank_0(f"F1: {metrics.get('f1', 0):.4f}", args.global_rank)
            print_rank_0(f"BLEU-2: {metrics.get('bleu2', 0):.4f}", args.global_rank)
            print_rank_0(f"ROUGE-1: {metrics.get('rouge1', 0):.4f}", args.global_rank)
            print_rank_0(f"ROUGE-2: {metrics.get('rouge2', 0):.4f}", args.global_rank)
            print_rank_0(f"Eval elapsed: {elapsed:.2f}s", args.global_rank)
            for ex in examples:
                print_rank_0(f"[EpochEndSample {ex['idx']}] REF: {ex['reference']} || PRED: {ex['prediction']}", args.global_rank)
            if args.global_rank <= 0:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(args.eval_log_file, 'a', encoding='utf-8') as lf:
                    lf.write(f"{now}\t{epoch+1}\t-1\tepoch_final\t" \
                             f"{metrics.get('precision','')}\t{metrics.get('recall','')}\t{metrics.get('f1','')}\t" \
                             f"{metrics.get('bleu2','')}\t{metrics.get('rouge1','')}\t{metrics.get('rouge2','')}\t{elapsed:.2f}\n")
            print_rank_0("=" * 60, args.global_rank)
            
            # 写入tensorboard和CSV (只在主进程执行)
            if args.global_rank <= 0:
                for metric_name, metric_value in metrics.items():
                    tb_write.add_scalar(f"eval_epoch/{metric_name}", metric_value, epoch + 1)
                save_metrics_to_csv(metrics, csv_path, epoch + 1, -1)

        if ds_config["zero_optimization"]["stage"] == 3:
            state_dict = model._zero3_consolidated_16bit_state_dict()
            if args.global_rank <= 0:
                save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}", state_dict)
        else:
            if args.global_rank <= 0:
                save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")


if __name__ == "__main__":
    main()
