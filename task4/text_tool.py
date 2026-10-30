#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量分析中文 .txt 文章，并生成单文件 report.html。

功能概览：
1. 读取指定文件夹中的全部 .txt 文章；
2. 使用 jieba 分词和 TF-IDF 提取每篇文章的 Top5 关键词；
3. 使用 TF-IDF 对句子打分，抽取每篇文章分数最高的 2 句作为核心句；
4. 生成带搜索、关键词屏蔽、恢复全部功能的单文件 HTML 报告。

分析逻辑位于 analyzer.py，命令行与网页应用共用。
依赖：jieba、scikit-learn。
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

# 分析逻辑集中在 analyzer.py，命令行和网页共用同一套函数。
from analyzer import Article, filter_articles, load_articles


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="批量提取中文文章关键词与核心句，并生成单文件 HTML 报告。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python text_tool.py samples\n"
            "  python text_tool.py samples --include 科技\n"
            "  python text_tool.py samples --exclude 广告 --output result.html\n"
            "  python text_tool.py samples --include 教育 --include 体育\n\n"
            "说明：--include 可重复使用，多个词之间为“同时包含”关系；\n"
            "      --exclude 命中任意一个词即排除。"
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(script_dir / "samples"),
        help="待分析的 .txt 文件夹（默认：脚本同级的 samples 目录）",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="词",
        help="只保留包含该词的文章，可重复使用",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="词",
        help="排除包含该词的文章，可重复使用",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "report.html"),
        help="输出 HTML 路径（默认：脚本同级的 report.html）",
    )
    return parser.parse_args(argv)


def render_keyword_tags(keywords: Sequence[Tuple[str, float]]) -> str:
    """渲染关键词标签，点击时由页面 JS 屏蔽包含该词的文章。"""
    if not keywords:
        return '<span class="muted">未提取到关键词</span>'

    tags: List[str] = []
    for index, (word, score) in enumerate(keywords):
        safe_word = html.escape(word, quote=True)
        tags.append(
            '<button type="button" class="tag tag-{color}" '
            'data-keyword="{word}" '
            'title="TF-IDF 权重 {score:.3f}；点击屏蔽包含“{word}”的文章" '
            'aria-label="屏蔽包含 {word} 的文章">{word}</button>'.format(
                color=index % 6,
                word=safe_word,
                score=score,
            )
        )
    return "".join(tags)


def render_core_sentences(sentences: Sequence[str]) -> str:
    """渲染核心句，使用 <mark> 做高亮，内容不改写。"""
    if not sentences:
        return '<p class="muted">未抽取到核心句</p>'

    blocks: List[str] = []
    for index, sentence in enumerate(sentences, start=1):
        blocks.append(
            '<div class="core-item">'
            '<span class="core-index">{index:02d}</span>'
            '<p class="core-text"><mark>{sentence}</mark></p>'
            "</div>".format(
                index=index,
                sentence=html.escape(sentence),
            )
        )
    return "".join(blocks)


def render_body(body: str) -> str:
    """把正文按段落渲染到“查看全文”区域。"""
    paragraphs = [line.strip() for line in body.split("\n") if line.strip()]
    return "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)


def render_card(article: Article, index: int) -> str:
    """渲染单篇文章卡片。"""
    search_text = " ".join(
        [article.title, article.body, " ".join(word for word, _ in article.keywords)]
    )
    safe_title = html.escape(article.title)
    safe_search = html.escape(search_text, quote=True)
    safe_filename = html.escape(article.path.name)

    return """
    <article class="card" data-search="{search}" aria-labelledby="card-title-{index}">
      <div class="card-top">
        <span class="card-no">{number:02d}</span>
        <span class="word-count">{count} 字</span>
      </div>
      <h2 class="card-title" id="card-title-{index}">{title}</h2>
      <p class="filename">{filename}</p>

      <div class="section-head">
        <span>关键词</span><span class="section-note">TF-IDF TOP 5</span>
      </div>
      <div class="tags">{tags}</div>

      <div class="section-head">
        <span>核心句</span><span class="section-note">抽取式 · 原文</span>
      </div>
      <div class="core-list">{cores}</div>

      <details class="fulltext">
        <summary>查看全文</summary>
        <div class="fulltext-body">{body}</div>
      </details>
    </article>
    """.format(
        search=safe_search,
        index=index,
        number=index,
        count=article.char_count,
        title=safe_title,
        filename=safe_filename,
        tags=render_keyword_tags(article.keywords),
        cores=render_core_sentences(article.core_sentences),
        body=render_body(article.body),
    )


def render_build_info(include_terms: Sequence[str], exclude_terms: Sequence[str]) -> str:
    """渲染构建时命令行筛选说明。"""
    parts: List[str] = []
    if include_terms:
        terms = "、".join(
            f"「{html.escape(term)}」" for term in include_terms
        )
        parts.append(f"包含 {terms}")
    if exclude_terms:
        terms = "、".join(
            f"「{html.escape(term)}」" for term in exclude_terms
        )
        parts.append(f"排除 {terms}")

    if not parts:
        return '<span class="build-dot"></span>构建筛选：无，展示文件夹内全部文章'
    return (
        '<span class="build-dot"></span>构建筛选：'
        + "；".join(parts)
        + ' <span class="build-hint">（页面内“恢复全部”不回退构建参数）</span>'
    )


def build_report_html(
    articles: Sequence[Article],
    source_folder: Path,
    include_terms: Sequence[str],
    exclude_terms: Sequence[str],
) -> str:
    """拼装最终的单文件 HTML。"""
    cards_html = "".join(
        render_card(article, index) for index, article in enumerate(articles, start=1)
    )

    # 先替换静态信息，最后插入卡片，避免用户文本中的占位符被误替换。
    page = HTML_TEMPLATE
    replacements = {
        "__TOTAL__": str(len(articles)),
        "__SOURCE__": html.escape(source_folder.name or str(source_folder)),
        "__GENERATED__": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "__BUILD_INFO__": render_build_info(include_terms, exclude_terms),
    }
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page.replace("__CARDS__", cards_html)


def print_analysis(articles: Sequence[Article]) -> None:
    """在控制台打印每篇文章的分析结果。"""
    total = len(articles)
    for index, article in enumerate(articles, start=1):
        print(f"\n[{index}/{total}] {article.title}")
        print(f"  文件：{article.path.name}")
        print(f"  字数：{article.char_count}")
        keyword_text = "  ".join(
            f"{word}({score:.3f})" for word, score in article.keywords
        )
        print(f"  关键词：{keyword_text or '无'}")
        print("  核心句：")
        for sentence in article.core_sentences:
            print(f"    - {sentence}")


def generate_report(
    articles: Sequence[Article],
    source_folder: Path,
    output_path: Path | str,
    include_terms: Sequence[str] = (),
    exclude_terms: Sequence[str] = (),
) -> Path:
    """生成 report.html 并返回实际输出路径。

    网页端与命令行都调用这里，保证报告模板、样式和筛选逻辑完全一致。
    """
    target = Path(output_path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)

    html_document = build_report_html(
        articles, source_folder, include_terms, exclude_terms
    )
    target.write_text(html_document, encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """主流程：解析参数 -> 读取文章 -> 分析 -> 筛选 -> 生成报告。"""
    args = parse_args(argv)
    folder = Path(args.folder).expanduser()

    if not folder.exists():
        print(f"[错误] 文件夹不存在：{folder}", file=sys.stderr)
        return 2
    if not folder.is_dir():
        print(f"[错误] 这不是文件夹：{folder}", file=sys.stderr)
        return 2

    print("=" * 64)
    print("中文文章关键词与核心句分析工具")
    print(f"输入目录：{folder.resolve()}")
    print(f"包含筛选：{'、'.join(args.include) if args.include else '无'}")
    print(f"排除筛选：{'、'.join(args.exclude) if args.exclude else '无'}")
    print("=" * 64)

    try:
        articles = load_articles(folder)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    print(f"\n已读取 {len(articles)} 篇有效文章，开始 TF-IDF 分析。")
    print_analysis(articles)

    filtered = filter_articles(articles, args.include, args.exclude)
    print("\n" + "-" * 64)
    print(f"筛选结果：{len(filtered)} / {len(articles)} 篇")
    if not filtered:
        print("[提示] 筛选后没有文章，未生成报告。", file=sys.stderr)
        return 1

    output_path = generate_report(
        filtered,
        folder,
        args.output,
        include_terms=args.include,
        exclude_terms=args.exclude,
    )

    print(f"报告已生成：{output_path.resolve()}")
    print(f"收录文章：{len(filtered)} 篇｜文件大小：{output_path.stat().st_size:,} 字节")
    print("=" * 64)
    return 0


# 单文件模板：CSS 与 JS 全部内联，不引用任何外部资源。
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>文章关键词与核心句 · 文本分析报告</title>
<style>
  :root {
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-soft: #f8faff;
    --text: #182230;
    --muted: #68758a;
    --line: #e4eaf2;
    --accent: #315fe0;
    --accent-soft: #edf2ff;
    --shadow: 0 16px 40px rgba(28, 48, 88, 0.08);
    --radius: 20px;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    min-height: 100vh;
    color: var(--text);
    background:
      radial-gradient(1100px 540px at 8% -8%, #e4ecff 0%, rgba(228, 236, 255, 0) 60%),
      radial-gradient(900px 500px at 100% 0%, #e7fbf7 0%, rgba(231, 251, 247, 0) 55%),
      var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
      "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  button, input { font: inherit; }
  button { color: inherit; }
  .page {
    width: min(1180px, calc(100% - 40px));
    margin: 0 auto;
    padding: 48px 0 72px;
  }
  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 7px 13px;
    border: 1px solid #d8e3f7;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.75);
    color: #3d5e9e;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
  }
  .eyebrow::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #36b78b;
    box-shadow: 0 0 0 4px rgba(54, 183, 139, 0.13);
  }
  h1 {
    max-width: 860px;
    margin: 18px 0 12px;
    font-size: clamp(30px, 4.6vw, 50px);
    line-height: 1.14;
    letter-spacing: -0.035em;
    font-weight: 760;
  }
  .subtitle {
    max-width: 760px;
    margin: 0;
    color: var(--muted);
    font-size: 16px;
  }
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 9px 20px;
    margin-top: 20px;
    color: #7a8799;
    font-size: 13px;
  }
  .meta span { display: inline-flex; align-items: center; gap: 7px; }
  .meta span::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #b6c2d6;
  }
  .panel {
    position: sticky;
    top: 14px;
    z-index: 20;
    margin: 30px 0 26px;
    padding: 16px;
    border: 1px solid rgba(222, 231, 243, 0.92);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
  }
  .search-row { display: flex; gap: 10px; }
  .search-box { position: relative; flex: 1; min-width: 0; }
  .search-icon {
    position: absolute;
    top: 50%;
    left: 17px;
    width: 19px;
    height: 19px;
    transform: translateY(-50%);
    color: #7f8ea4;
    pointer-events: none;
  }
  #search {
    width: 100%;
    height: 52px;
    padding: 0 48px 0 47px;
    border: 1px solid #dce5f1;
    border-radius: 13px;
    outline: none;
    background: #fbfdff;
    color: var(--text);
    font-size: 15px;
    transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
  }
  #search::placeholder { color: #98a5b7; }
  #search:focus {
    border-color: #8ca9ee;
    background: #fff;
    box-shadow: 0 0 0 4px rgba(49, 95, 224, 0.11);
  }
  .clear-search {
    position: absolute;
    top: 50%;
    right: 10px;
    width: 32px;
    height: 32px;
    padding: 0;
    transform: translateY(-50%);
    border: 0;
    border-radius: 9px;
    background: transparent;
    color: #7f8da2;
    font-size: 21px;
    line-height: 1;
    cursor: pointer;
  }
  .clear-search:hover { background: #eef3fa; color: #31415a; }
  .clear-search[hidden] { display: none; }
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 11px;
    padding: 0 2px;
  }
  #status {
    color: #53627a;
    font-size: 13px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .restore {
    flex: 0 0 auto;
    padding: 8px 14px;
    border: 1px solid #cedbf2;
    border-radius: 10px;
    background: #fff;
    color: #3156a6;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.18s, border-color 0.18s, transform 0.18s;
  }
  .restore:hover { background: var(--accent-soft); border-color: #aabfeb; }
  .restore:active { transform: translateY(1px); }
  .restore:disabled {
    cursor: default;
    opacity: 0.45;
    background: #f7f9fc;
  }
  .blocked-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-top: 11px;
    padding-top: 11px;
    border-top: 1px dashed #e1e8f2;
  }
  .blocked-bar[hidden] { display: none; }
  .blocked-label { color: #68758a; font-size: 12px; font-weight: 700; }
  .blocked-list { display: flex; flex-wrap: wrap; gap: 7px; }
  .blocked-chip {
    padding: 5px 10px;
    border: 1px solid #f0d4d8;
    border-radius: 999px;
    background: #fff5f6;
    color: #a23a49;
    font-size: 12px;
    font-weight: 650;
    cursor: pointer;
  }
  .blocked-chip:hover { background: #ffe9ec; border-color: #e9b8c0; }
  .build-info {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-top: 10px;
    color: #7a8799;
    font-size: 12px;
  }
  .build-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #b8c6da;
  }
  .build-hint { color: #99a5b5; }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    align-items: start;
  }
  .card {
    position: relative;
    display: flex;
    flex-direction: column;
    padding: 22px;
    overflow: hidden;
    border: 1px solid #e4eaf3;
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 8px 26px rgba(31, 50, 85, 0.05);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
  }
  .card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #5d82ef, #49bfa3 55%, #f2b64f);
    opacity: 0.82;
  }
  .card:hover {
    transform: translateY(-3px);
    border-color: #d4dfef;
    box-shadow: 0 17px 38px rgba(31, 50, 85, 0.09);
  }
  .card[hidden] { display: none !important; }
  .card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 11px;
  }
  .card-no {
    color: #a3b0c3;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.14em;
    font-variant-numeric: tabular-nums;
  }
  .word-count {
    padding: 4px 9px;
    border: 1px solid #e5ebf4;
    border-radius: 999px;
    background: #f8fafd;
    color: #6f7d91;
    font-size: 12px;
    font-weight: 650;
    white-space: nowrap;
  }
  .card-title {
    margin: 0 0 6px;
    font-size: 21px;
    line-height: 1.42;
    letter-spacing: -0.015em;
    font-weight: 750;
  }
  .filename {
    margin: 0;
    color: #99a5b6;
    font-size: 11.5px;
    word-break: break-all;
  }
  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 19px 0 9px;
    color: #40506a;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
  }
  .section-note {
    color: #a3afbf;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
  }
  .tags { display: flex; flex-wrap: wrap; gap: 7px; }
  .tag {
    padding: 6px 11px;
    border: 1px solid transparent;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 700;
    line-height: 1.2;
    cursor: pointer;
    transition: transform 0.16s, filter 0.16s, box-shadow 0.16s;
  }
  .tag:hover {
    transform: translateY(-1px);
    filter: saturate(1.15);
    box-shadow: 0 4px 10px rgba(31, 50, 85, 0.08);
  }
  .tag:focus-visible {
    outline: 3px solid rgba(49, 95, 224, 0.25);
    outline-offset: 2px;
  }
  .tag-0 { background: #eaf1ff; border-color: #d6e3ff; color: #2855bd; }
  .tag-1 { background: #e8f8f2; border-color: #d0efe4; color: #14765c; }
  .tag-2 { background: #fff7e4; border-color: #f6e7bf; color: #8a6414; }
  .tag-3 { background: #fff0f2; border-color: #f5d9dd; color: #a23e4d; }
  .tag-4 { background: #f2efff; border-color: #e2dcfb; color: #5b44a9; }
  .tag-5 { background: #e9f7fb; border-color: #d0ebf2; color: #14687b; }
  .core-list { display: grid; gap: 9px; }
  .core-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 13px;
    border: 1px solid #f3e7c6;
    border-radius: 12px;
    background: #fffdf5;
  }
  .core-index {
    flex: 0 0 auto;
    padding-top: 2px;
    color: #c29a3b;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.06em;
    font-variant-numeric: tabular-nums;
  }
  .core-text {
    margin: 0;
    color: #4d4636;
    font-size: 14px;
    line-height: 1.85;
  }
  mark {
    padding: 0 2px;
    color: inherit;
    background: linear-gradient(transparent 54%, #ffe389 54%);
  }
  .fulltext {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px dashed #e2e9f2;
  }
  .fulltext summary {
    width: fit-content;
    color: #65748a;
    font-size: 12.5px;
    font-weight: 700;
    cursor: pointer;
    list-style: none;
    user-select: none;
  }
  .fulltext summary::-webkit-details-marker { display: none; }
  .fulltext summary::before {
    content: "＋";
    display: inline-block;
    margin-right: 6px;
    color: #8da0bc;
    font-weight: 700;
    transition: transform 0.18s;
  }
  .fulltext[open] summary::before { content: "－"; }
  .fulltext-body {
    margin-top: 10px;
    padding: 13px 15px;
    border-radius: 12px;
    background: #f8fafd;
  }
  .fulltext-body p {
    margin: 0 0 10px;
    color: #536176;
    font-size: 13.5px;
    line-height: 1.9;
  }
  .fulltext-body p:last-child { margin-bottom: 0; }
  .empty {
    padding: 64px 20px;
    border: 1px dashed #cfdaea;
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.6);
    text-align: center;
  }
  .empty[hidden] { display: none; }
  .empty-icon {
    display: grid;
    place-items: center;
    width: 58px;
    height: 58px;
    margin: 0 auto 15px;
    border-radius: 50%;
    background: #edf2fa;
    color: #7e90aa;
    font-size: 25px;
  }
  .empty h2 { margin: 0 0 7px; font-size: 18px; }
  .empty p { margin: 0; color: var(--muted); font-size: 13.5px; }
  .footer {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px 20px;
    margin-top: 32px;
    padding-top: 18px;
    border-top: 1px solid #dfe7f1;
    color: #8b98ab;
    font-size: 12px;
  }
  .muted { color: #98a4b5; font-size: 13px; }
  @media (max-width: 840px) {
    .page { width: min(100% - 28px, 720px); padding-top: 32px; }
    .grid { grid-template-columns: 1fr; }
    .panel { position: static; }
    .card-title { font-size: 19px; }
  }
  @media (max-width: 520px) {
    .page { width: min(100% - 22px, 520px); padding: 24px 0 48px; }
    .panel { padding: 12px; border-radius: 15px; }
    .search-row { gap: 0; }
    #search { height: 48px; font-size: 14px; }
    .toolbar { align-items: flex-start; }
    .card { padding: 18px; border-radius: 16px; }
    .card-title { font-size: 18px; }
    .meta { font-size: 12px; }
    .footer { display: block; line-height: 1.9; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      scroll-behavior: auto !important;
      transition: none !important;
    }
  }
</style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">TEXT INSIGHT · 文本分析报告</div>
      <h1>文章关键词与核心句</h1>
      <p class="subtitle">基于 jieba 分词与 TF-IDF 权重，从每篇中文文章中提取 5 个关键词和 2 句核心句。点击关键词可屏蔽包含该词的文章，搜索框支持实时过滤。</p>
      <div class="meta">
        <span>共 <strong>__TOTAL__</strong> 篇</span>
        <span>来源：__SOURCE__</span>
        <span>生成于 __GENERATED__</span>
      </div>
    </header>

    <section class="panel" aria-label="报告筛选">
      <div class="search-row">
        <label class="search-box">
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5" stroke="currentColor" stroke-width="1.8"></circle>
            <path d="M16 16L21 21" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
          </svg>
          <input id="search" type="search" autocomplete="off" spellcheck="false"
            placeholder="搜索标题、正文或关键词…" aria-label="搜索文章">
          <button type="button" id="clearSearch" class="clear-search" hidden
            aria-label="清空搜索">×</button>
        </label>
      </div>
      <div class="toolbar">
        <div id="status" role="status" aria-live="polite">显示 __TOTAL__ / __TOTAL__ 篇</div>
        <button type="button" id="restoreAll" class="restore">恢复全部</button>
      </div>
      <div id="blockedBar" class="blocked-bar" hidden>
        <span class="blocked-label">屏蔽中：</span>
        <span id="blockedList" class="blocked-list"></span>
      </div>
      <div class="build-info">__BUILD_INFO__</div>
    </section>

    <main id="grid" class="grid" aria-label="文章列表">
__CARDS__
    </main>

    <section id="empty" class="empty" hidden>
      <div class="empty-icon">⌕</div>
      <h2>没有匹配的文章</h2>
      <p>换个关键词试试，或点击“恢复全部”回到初始状态。</p>
    </section>

    <footer class="footer">
      <span>本报告由 text_tool.py 生成 · jieba + scikit-learn TF-IDF</span>
      <span>单文件离线报告 · 无外部资源</span>
    </footer>
  </div>

<script>
(function () {
  "use strict";

  var cards = Array.prototype.slice.call(document.querySelectorAll(".card"));
  var searchInput = document.getElementById("search");
  var clearButton = document.getElementById("clearSearch");
  var restoreButton = document.getElementById("restoreAll");
  var status = document.getElementById("status");
  var blockedBar = document.getElementById("blockedBar");
  var blockedList = document.getElementById("blockedList");
  var emptyState = document.getElementById("empty");
  var grid = document.getElementById("grid");

  // 键为小写词，值为展示用原词；用 Map 避免大小写重复。
  var blocked = new Map();

  function normalize(value) {
    return String(value || "").toLocaleLowerCase("zh-CN").trim();
  }

  function articleContains(card, term) {
    return normalize(card.dataset.search).indexOf(normalize(term)) !== -1;
  }

  function makeBlockedChip(key, term) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "blocked-chip";
    button.textContent = term + " ×";
    button.title = "解除屏蔽：" + term;
    button.setAttribute("aria-label", "解除屏蔽：" + term);
    button.addEventListener("click", function () {
      blocked.delete(key);
      render();
    });
    return button;
  }

  function render() {
    var query = normalize(searchInput.value);
    var blockedEntries = Array.from(blocked.entries());
    var visible = 0;

    cards.forEach(function (card) {
      var queryOK = !query || normalize(card.dataset.search).indexOf(query) !== -1;
      var blockedOK = !blockedEntries.some(function (entry) {
        return articleContains(card, entry[1]);
      });
      var show = queryOK && blockedOK;
      card.hidden = !show;
      if (show) visible += 1;
    });

    var statusParts = ["显示 " + visible + " / " + cards.length + " 篇"];
    if (query) statusParts.push("搜索“" + searchInput.value.trim() + "”");
    if (blocked.size) statusParts.push("屏蔽 " + blocked.size + " 个词");
    status.textContent = statusParts.join(" · ");

    clearButton.hidden = !query;
    emptyState.hidden = visible !== 0;
    blockedBar.hidden = blocked.size === 0;
    restoreButton.disabled = !query && blocked.size === 0;

    blockedList.textContent = "";
    blockedEntries.forEach(function (entry) {
      blockedList.appendChild(makeBlockedChip(entry[0], entry[1]));
    });
  }

  // 点击任意关键词标签：屏蔽包含该词的全部文章。
  grid.addEventListener("click", function (event) {
    var tag = event.target.closest(".tag");
    if (!tag) return;
    var term = tag.getAttribute("data-keyword");
    if (!term) return;
    blocked.set(normalize(term), term);
    render();
  });

  searchInput.addEventListener("input", render);
  searchInput.addEventListener("search", render);
  searchInput.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      searchInput.value = "";
      render();
    }
  });

  clearButton.addEventListener("click", function () {
    searchInput.value = "";
    render();
    searchInput.focus();
  });

  // 恢复全部：清空搜索词与所有关键词屏蔽，但不改变构建时的 --include/--exclude。
  restoreButton.addEventListener("click", function () {
    searchInput.value = "";
    blocked.clear();
    render();
    searchInput.focus();
  });

  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
