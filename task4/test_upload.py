#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多文件上传自动测试。

用 requests 在一个 multipart/form-data 请求中同时 POST 3 个 .txt 文件，
对比上传前后的 index.json 篇数，并逐个确认 3 个文件名都进入了索引。

用法：
    python test_upload.py              # 测试后自动删除本次测试文献
    python test_upload.py --keep       # 保留测试文献，方便在网页中查看
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.json"
SAMPLES_DIR = BASE_DIR / "samples"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"


def parse_args() -> argparse.Namespace:
    """解析测试参数。"""
    parser = argparse.ArgumentParser(description="验证 /upload 是否真正支持多文件。")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"服务地址（默认：{DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="测试完成后不删除 3 篇测试文献",
    )
    return parser.parse_args()


def load_index() -> List[Dict[str, Any]]:
    """读取当前 index.json。"""
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def build_test_files() -> Tuple[List[Tuple[str, Tuple[str, bytes, str]]], List[str]]:
    """从样例文章生成 3 个文件名唯一的测试 .txt。"""
    samples = sorted(SAMPLES_DIR.glob("*.txt"))[:3]
    if len(samples) < 3:
        raise RuntimeError("samples/ 中至少需要 3 篇 .txt 才能运行测试。")

    stamp = datetime.now().strftime("%H%M%S")
    token = uuid.uuid4().hex[:4]
    payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
    filenames: List[str] = []

    for index, sample in enumerate(samples, start=1):
        filename = f"test_multi_{stamp}_{token}_{index}.txt"
        content = sample.read_bytes()
        payload.append(
            ("files", (filename, content, "text/plain; charset=utf-8"))
        )
        filenames.append(filename)
    return payload, filenames


def main() -> int:
    """执行上传、校验和可选清理。"""
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    payload, filenames = build_test_files()

    before_entries = load_index()
    before_count = len(before_entries)
    print("=" * 62)
    print("多文件上传测试")
    print(f"服务地址：{base_url}")
    print(f"上传前 index.json 篇数：{before_count}")
    print("本次一次 POST 的 3 个文件：")
    for filename in filenames:
        print(f"  - {filename}")

    created_ids: List[str] = []
    try:
        with requests.Session() as session:
            response = session.post(
                f"{base_url}/upload",
                files=payload,
                timeout=60,
                allow_redirects=True,
            )
            response.raise_for_status()

        after_entries = load_index()
        after_count = len(after_entries)
        uploaded = [
            entry for entry in after_entries if entry.get("filename") in filenames
        ]
        created_ids = [str(entry.get("id")) for entry in uploaded]

        print("-" * 62)
        print(f"POST /upload 状态码：{response.status_code}")
        print(f"上传后 index.json 篇数：{after_count}")
        print(f"本次新增篇数：{len(uploaded)}")
        print("新增记录：")
        for entry in uploaded:
            keywords = "、".join(entry.get("keywords", []))
            print(
                f"  - {entry.get('filename')} | id={entry.get('id')} | "
                f"{entry.get('word_count')} 字 | {entry.get('title')}"
            )
            print(f"    关键词：{keywords}")

        # 断言：3 个文件必须同时进入索引，证明不是只处理了第一个。
        if len(uploaded) != 3:
            print(f"[失败] 期望新增 3 篇，实际新增 {len(uploaded)} 篇。")
            return 1
        if after_count != before_count + 3:
            print(
                f"[失败] 期望总篇数 {before_count + 3}，实际 {after_count}。"
            )
            return 1
        if "成功 3 篇，跳过 0 篇" not in response.text:
            print("[提示] 页面未找到预期的成功提示，请检查 /upload 返回页面。")
        else:
            print("[通过] 页面提示：成功 3 篇，跳过 0 篇。")

        print("[通过] 3 个文件均在 index.json 中，说明一次请求上传了全部文件。")
        return 0
    finally:
        if created_ids and not args.keep:
            with requests.Session() as session:
                for article_id in created_ids:
                    try:
                        session.post(
                            f"{base_url}/delete/{article_id}",
                            timeout=20,
                            allow_redirects=False,
                        )
                    except requests.RequestException as exc:
                        print(f"[警告] 清理 {article_id} 失败：{exc}")
            print(f"清理后 index.json 篇数：{len(load_index())}")
        elif created_ids and args.keep:
            print("已保留 3 篇测试文献，可在网页中查看后手动删除。")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as exc:
        print(f"[错误] 无法访问服务：{exc}", file=sys.stderr)
        print("请先运行：python app.py", file=sys.stderr)
        sys.exit(2)
