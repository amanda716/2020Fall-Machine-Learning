"""Minimal Qwen caller using OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def call_qwen(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
) -> str:
    """Call Qwen chat completion and return plain text output."""
    if OpenAI is None:
        raise ImportError("未安装 openai 依赖，请先执行: pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简洁的AI助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _resolve_api_key(cli_api_key: Optional[str]) -> str:
    key = cli_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("缺少 API Key：请传入 --api-key 或设置环境变量 DASHSCOPE_API_KEY")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="调用千问模型（qwen-plus）")
    parser.add_argument("--prompt", default="请用一句话介绍机器学习。", help="用户输入提示词")
    parser.add_argument("--api-key", default=None, help="可选：直接传 API Key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称，默认 qwen-plus")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 兼容接口 Base URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    args = parser.parse_args()

    api_key = _resolve_api_key(args.api_key)
    output = call_qwen(
        prompt=args.prompt,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    print(output)


if __name__ == "__main__":
    main()
