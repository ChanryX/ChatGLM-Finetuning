"""ModelScope 模型下载脚本

支持:
1. 指定模型 ID
2. 指定保存目录 (local_dir)
3. 指定缓存根目录 (cache_dir)
4. Linux 默认下载到 /root/autodl-fs/chatglm-6b (若未传参且无环境变量)
5. 通过环境变量 MODEL_DOWNLOAD_DIR 覆盖默认保存目录

示例:
  python download,py --model-id ZhipuAI/ChatGLM-6B --dest /root/autodl-fs/chatglm6b
或只执行 (Linux 上会自动放到 /root/autodl-fs/chatglm-6b):
  python download,py
"""

from modelscope import snapshot_download
import argparse
from pathlib import Path
import os


def default_dest() -> str:
    env_path = os.environ.get("MODEL_DOWNLOAD_DIR")
    if env_path:
        return env_path
    if os.name == "posix":  # Linux/Unix
        return "/root/autodl-fs/chatglm-6b"
    return "./models/chatglm-6b"


def parse_args():
    p = argparse.ArgumentParser(description="Download ModelScope model to a specified directory")
    p.add_argument("--model-id", default="ZhipuAI/ChatGLM-6B", help="模型 ID (org/name)")
    p.add_argument("--dest", default=default_dest(), help="保存目录，可用环境变量 MODEL_DOWNLOAD_DIR 覆盖")
    p.add_argument("--cache-root", default=None, help="缓存根目录 cache_dir")
    p.add_argument("--revision", default=None, help="指定版本/分支/commit (不填则最新)")
    return p.parse_args()


def main():
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"开始下载: model_id={args.model_id}")
    print(f"保存目录(local_dir): {dest}")
    if args.cache_root:
        print(f"缓存根(cache_dir): {args.cache_root}")
    if args.revision:
        print(f"指定 revision: {args.revision}")

    model_dir = snapshot_download(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_root,
        local_dir=str(dest),
    )
    print(f"模型已下载到: {model_dir}")
    if str(dest).startswith('/root/autodl-fs'):
        print("提示: 确认 /root/autodl-fs 所在磁盘空间是否足够 (ChatGLM-6B ~14G)。")


if __name__ == "__main__":
    main()