# -*- coding:utf-8 -*-
"""独立评估脚本：支持信息抽取(ie)与问答(qa)任务；计算 Precision/Recall/F1 (仅ie)、BLEU-2、ROUGE-1/2。
使用时配合run_eval.sh
"""
import argparse
import json
import math
import os
from collections import Counter
import torch
import time
from model import MODE
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ================= 基础函数 =================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return s.replace("\r", "").replace("\t", "").strip()

def extract_triplet_lines(text: str):
    if not text:
        return []
    lines = [ln.strip() for ln in text.split('\n')]
    return [ln for ln in lines if ln and '_' in ln]

def compute_ie_counts(pred_lines, ref_lines):
    pset, rset = set(pred_lines), set(ref_lines)
    tp = len(pset & rset)
    fp = len(pset - rset)
    fn = len(rset - pset)
    return tp, fp, fn

def _get_char_ngrams(s: str, n: int):
    s = normalize_text(s).replace(' ', '').replace('\n', '')
    return Counter([s[i:i+n] for i in range(0, max(len(s)-n+1, 0))])

def compute_corpus_bleu_cumulative(pred_texts, ref_texts, max_n: int = 2) -> float:
    if not pred_texts or not ref_texts:
        return 0.0
    total_pred_len = 0
    total_ref_len = 0
    p_ns = []
    for n in range(1, max_n + 1):
        overlap = 0
        pred_total = 0
        for pred, ref in zip(pred_texts, ref_texts):
            p_ng = _get_char_ngrams(pred, n)
            r_ng = _get_char_ngrams(ref, n)
            pred_total += sum(p_ng.values())
            for g, c in p_ng.items():
                overlap += min(c, r_ng.get(g, 0))
        p_n = overlap / pred_total if pred_total > 0 else 0.0
        p_ns.append(p_n)
    for pred, ref in zip(pred_texts, ref_texts):
        total_pred_len += len(normalize_text(pred).replace(' ', '').replace('\n', ''))
        total_ref_len += len(normalize_text(ref).replace(' ', '').replace('\n', ''))
    if total_pred_len == 0:
        return 0.0
    if total_pred_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (total_ref_len / max(total_pred_len, 1)))
    eps = 1e-12
    return bp * math.exp(sum(math.log(max(p, eps)) for p in p_ns) / len(p_ns))

def compute_corpus_rouge_n(pred_texts, ref_texts, n: int) -> float:
    if not pred_texts or not ref_texts:
        return 0.0
    overlap = 0
    ref_total = 0
    for pred, ref in zip(pred_texts, ref_texts):
        p_ng = _get_char_ngrams(pred, n)
        r_ng = _get_char_ngrams(ref, n)
        ref_total += sum(r_ng.values())
        for g, c in r_ng.items():
            overlap += min(c, p_ng.get(g, 0))
    return overlap / ref_total if ref_total > 0 else 0.0

# ================= 评估主流程 =================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True, type=str, help='模型或checkpoint目录')
    ap.add_argument('--test_path', required=True, type=str, help='测试数据 jsonl')
    ap.add_argument('--mode', type=str, default='glm', choices=['glm','glm2','glm3'])
    ap.add_argument('--task_type', type=str, default='ie', choices=['ie','qa'])
    ap.add_argument('--device', type=str, default='0', help="CUDA 设备，如 '0' 或 '0,1' 或 'cpu'")
    ap.add_argument('--max_length', type=int, default=1024, help='生成时允许的最大总长度(提示+输出)硬上限')
    ap.add_argument('--max_new_tokens', type=int, default=256, help='生成允许的新token上限 (会与prompt长度一起裁剪不超过 max_length)')
    ap.add_argument('--eval_samples', type=int, default=1000, help='最大评估样本数')
    ap.add_argument('--do_sample', action='store_true')
    ap.add_argument('--top_p', type=float, default=0.8)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--save_predictions', type=str, default=None, help='保存预测 jsonl 路径')
    ap.add_argument('--base_model_path', type=str, default=None, help='(LoRA) 基础模型目录; 仅在目录为纯 adapter(无 config.json) 时需要，用于自动合并。')
    return ap.parse_args()

def main():
    args = parse_args()
    use_cpu = args.device.lower() == 'cpu' or not torch.cuda.is_available()
    device_ids = [] if use_cpu else [int(args.device.split(',')[0])]
    device = torch.device('cpu' if use_cpu else f'cuda:{device_ids[0]}')

    # ================= LoRA / 普通模型加载与(可选)合并 =================
    adapter_config_file = os.path.join(args.model_path, 'adapter_config.json')
    load_dtype = torch.float32 if use_cpu else torch.float16
    tokenizer = None

    has_full = os.path.exists(os.path.join(args.model_path, 'config.json'))
    has_adapter = os.path.exists(adapter_config_file)
    is_pure_adapter = has_adapter and not has_full

    if is_pure_adapter:
        if not args.base_model_path:
            raise RuntimeError('检测到 LoRA 适配器目录但未提供 --base_model_path，无法合并生成完整模型。')
        if PeftModel is None:
            raise RuntimeError('需要 peft 包来加载 LoRA 适配器。')
        print(f'[LoRA] 发现纯适配器目录: {args.model_path} ，开始加载基础模型并合并 (保留 adapter_model.bin)')
        base_model = MODE[args.mode]['model'].from_pretrained(args.base_model_path, torch_dtype=load_dtype)
        lora_model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=load_dtype)
        model = lora_model.merge_and_unload()
        # 保存完整模型到同级目录（不会删除 adapter 文件）
        print(f'[LoRA-Merge] 写入完整模型权重到原目录: {args.model_path}')
        model.save_pretrained(args.model_path, max_shard_size='2GB')
        # 保存 tokenizer （只在首次）
        try:
            MODE[args.mode]['tokenizer'].from_pretrained(args.base_model_path).save_pretrained(args.model_path)
        except Exception as te:
            print(f'[LoRA-Merge] Tokenizer 保存失败(忽略): {te}')
        tokenizer = MODE[args.mode]['tokenizer'].from_pretrained(args.model_path)
    else:
        # 已含完整模型（可能同时含 adapter），直接当完整模型加载
        load_from = args.model_path
        model = MODE[args.mode]['model'].from_pretrained(load_from, torch_dtype=load_dtype)
        tokenizer = MODE[args.mode]['tokenizer'].from_pretrained(load_from)
        if has_adapter and not args.base_model_path:
            print('[LoRA Info] 目录包含 adapter 与完整模型，已直接使用完整模型；若需重新合并请删除 config.json 后再运行。')
    
    model.to(device)
    model.eval()

    predictions = []
    references = []
    micro_tp = micro_fp = micro_fn = 0

    save_f = open(args.save_predictions, 'w', encoding='utf-8') if args.save_predictions else None

    # 读取所有行（保留顺序）
    with open(args.test_path, 'r', encoding='utf-8') as fh:
        raw_lines = [ln for ln in fh if ln.strip()]
    total_lines = min(len(raw_lines), args.eval_samples)
    iterator = range(total_lines)
    if tqdm is not None:
        iterator = tqdm(iterator, desc='Evaluating', ncols=80)
    last_print = time.time()
    for idx in iterator:
        line = raw_lines[idx]
        try:
            sample = json.loads(line.strip())
        except Exception:
            continue
        instruction = sample.get('instruction', '')
        _input = sample.get('input', '')
        ref = normalize_text(sample.get('output', ''))
        prompt = f"{instruction}{_input}"
        with torch.no_grad():
            # 动态计算max_length，避免截断prompt本身
            try:
                prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                prompt_len = prompt_ids.shape[-1]
            except Exception:
                prompt_len = 0
            dynamic_max_length = min(args.max_length, prompt_len + args.max_new_tokens)
            gen_kwargs = dict(max_length=dynamic_max_length)
            if args.do_sample:
                gen_kwargs.update(dict(do_sample=True, top_p=args.top_p, temperature=args.temperature))
            else:
                gen_kwargs.update(dict(do_sample=False))
            try:
                result, _ = model.chat(tokenizer, prompt, **gen_kwargs)
                pred = normalize_text(result)
            except Exception as e:
                pred = ''
        predictions.append(pred)
        references.append(ref)
        if args.task_type == 'ie':
            pred_lines = extract_triplet_lines(pred)
            ref_lines = extract_triplet_lines(ref)
            tp, fp, fn = compute_ie_counts(pred_lines, ref_lines)
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
        if save_f:
            save_f.write(json.dumps({'instruction': instruction, 'input': _input, 'reference': ref, 'prediction': pred}, ensure_ascii=False) + '\n')
        if tqdm is None and (time.time() - last_print) > 5:
            print(f'[Progress] {idx+1}/{total_lines}')
            last_print = time.time()
    if save_f:
        save_f.close()

    metrics = {}
    if args.task_type == 'ie':
        precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.update(dict(precision=precision, recall=recall, f1=f1))
    metrics['bleu2'] = compute_corpus_bleu_cumulative(predictions, references, 2)
    metrics['rouge1'] = compute_corpus_rouge_n(predictions, references, 1)
    metrics['rouge2'] = compute_corpus_rouge_n(predictions, references, 2)

    print('=' * 60)
    print(f"Samples evaluated: {len(predictions)}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    if args.save_predictions:
        print(f"Predictions saved to: {os.path.abspath(args.save_predictions)}")
    print('=' * 60)

if __name__ == '__main__':
    main()
