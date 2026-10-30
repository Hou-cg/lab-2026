#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文献库 Web 管理应用。

功能：
1. 上传多个 .txt 文献到 library/，立即分析并写入 index.json；
2. 在网页中查看标题、字数、关键词、导入时间；
3. 删除文献时同步删除文件和索引记录；
4. 一键用 library/ 中的全部文献重新生成 report.html。

启动方式：
    python app.py
然后访问 http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import re
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

import analyzer
import text_tool

# 文献库与索引文件均放在 task4/ 下，和 templates/ 同级。
BASE_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = BASE_DIR / "library"
INDEX_FILE = BASE_DIR / "index.json"
REPORT_FILE = BASE_DIR / "report.html"
DATE_FORMAT = "%Y-%m-%d %H:%M"

# Windows 文件名中不允许出现的字符；保留中文、空格和常见符号。
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 单次上传最多 80 MB


def ensure_storage() -> None:
    """确保 library/ 和 index.json 存在。"""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        save_index([])


def load_index() -> List[Dict[str, Any]]:
    """读取 index.json；文件损坏或格式错误时返回空列表。"""
    if not INDEX_FILE.exists():
        return []
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.warning("读取 index.json 失败：%s", exc)
        return []

    if not isinstance(data, list):
        app.logger.warning("index.json 顶层不是列表，已按空文献库处理。")
        return []

    entries: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("id") or not item.get("filename"):
            continue
        entries.append(item)
    return entries


def save_index(entries: Sequence[Dict[str, Any]]) -> None:
    """原子写入 index.json，避免写入中断留下半个文件。"""
    safe_entries = list(entries)
    temporary = INDEX_FILE.with_name(INDEX_FILE.name + ".tmp")
    temporary.write_text(
        json.dumps(safe_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(INDEX_FILE)


def new_id(existing_ids: Sequence[str]) -> str:
    """生成不与现有记录冲突的 8 位文章 ID。"""
    used = set(existing_ids)
    while True:
        candidate = uuid.uuid4().hex[:8]
        if candidate not in used:
            return candidate


def entry_from_article(
    article: analyzer.Article,
    article_id: str,
    added: str,
) -> Dict[str, Any]:
    """把分析结果转换成 index.json 中的一条元数据。"""
    return {
        "id": article_id,
        "filename": article.path.name,
        "title": article.title,
        "word_count": article.char_count,
        "keywords": [word for word, _ in article.keywords],
        "added": added,
    }


def sync_index(
    entries: Sequence[Dict[str, Any]], articles: Sequence[analyzer.Article]
) -> List[Dict[str, Any]]:
    """让索引与 library/ 的实际文件保持一致。

    - 已有记录保留原 id 与导入时间，只刷新标题、字数、关键词；
    - 新出现的文件自动补一条记录；
    - 文件已不存在的记录会被移除。
    """
    by_filename = {article.path.name: article for article in articles}
    known_ids = [str(item.get("id")) for item in entries if item.get("id")]
    now = datetime.now().strftime(DATE_FORMAT)
    result: List[Dict[str, Any]] = []
    seen_filenames = set()

    for item in entries:
        filename = item.get("filename")
        if not isinstance(filename, str) or filename not in by_filename:
            continue
        article = by_filename[filename]
        article_id = str(item.get("id") or new_id(known_ids))
        if article_id not in known_ids:
            known_ids.append(article_id)
        result.append(
            entry_from_article(
                article,
                article_id,
                str(item.get("added") or now),
            )
        )
        seen_filenames.add(filename)

    for filename, article in sorted(by_filename.items()):
        if filename in seen_filenames:
            continue
        article_id = new_id(known_ids)
        known_ids.append(article_id)
        result.append(entry_from_article(article, article_id, now))

    return result


def analyze_library() -> List[analyzer.Article]:
    """分析 library/ 中的全部 .txt；空文件夹返回空列表。"""
    if not any(LIBRARY_DIR.glob("*.txt")):
        return []
    return analyzer.load_articles(LIBRARY_DIR)


def refresh_library_index(
    entries: Sequence[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], List[analyzer.Article]]:
    """重新分析文献库，更新并保存 index.json。"""
    current = list(entries) if entries is not None else load_index()
    articles = analyze_library()
    synced = sync_index(current, articles)
    save_index(synced)
    return synced, articles


def safe_upload_filename(original: str) -> str:
    """清理上传文件名并保留中文，不自动改名。

    同名冲突由上传流程判断并跳过，绝不以新文件覆盖已有文献。
    """
    raw_name = (original or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if raw_name.lower().endswith(".txt"):
        raw_name = raw_name[:-4]
    stem = INVALID_FILENAME_CHARS.sub("_", raw_name).strip(" .")
    stem = stem[:80] or "未命名"
    if stem.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = f"_{stem}"
    return f"{stem}.txt"


def safe_library_path(filename: str) -> Path | None:
    """把索引中的文件名限制在 library/ 内，防止路径穿越。"""
    if not filename or Path(filename).name != filename:
        return None
    path = (LIBRARY_DIR / filename).resolve()
    if path.parent != LIBRARY_DIR.resolve():
        return None
    if path.suffix.lower() != ".txt":
        return None
    return path


def skipped_message(skipped: Sequence[str]) -> str:
    """把跳过的文件压缩成一条简短提示。"""
    preview = "；".join(skipped[:3])
    if len(skipped) > 3:
        preview += f"；另有 {len(skipped) - 3} 个文件"
    return f"已跳过：{preview}"


@app.get("/")
def index():
    """文献库首页：读取 index.json 并展示文章列表。"""
    entries = load_index()
    entries.sort(key=lambda item: str(item.get("added", "")), reverse=True)
    total_words = sum(int(item.get("word_count") or 0) for item in entries)
    last_added = str(entries[0].get("added", "")) if entries else ""
    return render_template(
        "index.html",
        articles=entries,
        total_words=total_words,
        last_added=last_added,
    )


@app.post("/upload")
def upload():
    """接收多个 .txt 文件：逐个保存、解析，再统一分析并写入索引。"""
    uploaded_files = [
        storage
        for storage in request.files.getlist("files")
        if storage and storage.filename
    ]
    if not uploaded_files:
        flash("请选择至少一个 .txt 文件。", "error")
        return redirect(url_for("index"))

    entries = load_index()
    saved: List[str] = []
    skipped: List[str] = []
    # 同名判断不区分大小写；同一批次内也使用这份集合防止互相覆盖。
    existing_names = {
        path.name.casefold() for path in LIBRARY_DIR.glob("*.txt")
    }

    for storage in uploaded_files:
        original = storage.filename or ""
        if not original.lower().endswith(".txt"):
            skipped.append(f"{original}（不是 .txt 文件）")
            continue

        filename = safe_upload_filename(original)
        if filename.casefold() in existing_names:
            skipped.append(f"{original}（同名文件已存在）")
            continue

        path = LIBRARY_DIR / filename
        if path.exists():
            skipped.append(f"{original}（同名文件已存在）")
            continue

        try:
            data = storage.read()
            if not data.strip():
                skipped.append(f"{original}（空文件）")
                continue
            path.write_bytes(data)
            existing_names.add(filename.casefold())

            # 立即解析，确保标题和正文有效；无效文件从文献库移除。
            article = analyzer.parse_article(path)
            if article is None:
                path.unlink(missing_ok=True)
                existing_names.discard(filename.casefold())
                skipped.append(f"{original}（缺少标题或正文）")
                continue
        except (UnicodeDecodeError, OSError) as exc:
            path.unlink(missing_ok=True)
            existing_names.discard(filename.casefold())
            app.logger.warning("读取上传文件 %s 失败：%s", original, exc)
            skipped.append(f"{original}（编码或写入错误）")
            continue

        # 先追加基础元数据；整批保存完成后统一做全库 TF-IDF 分析并刷新关键词。
        entries.append(
            {
                "id": new_id([str(item.get("id", "")) for item in entries]),
                "filename": filename,
                "title": article.title,
                "word_count": article.char_count,
                "keywords": [],
                "added": datetime.now().strftime(DATE_FORMAT),
            }
        )
        saved.append(filename)

    if saved:
        try:
            entries, _ = refresh_library_index(entries)
            # 文献库发生变化后，旧报告不再对应当前内容。
            REPORT_FILE.unlink(missing_ok=True)
            flash(
                f"导入完成：成功 {len(saved)} 篇，跳过 {len(skipped)} 篇。",
                "success" if not skipped else "info",
            )
        except Exception as exc:  # 分析依赖或文件异常时给出可读提示
            app.logger.exception("导入后分析失败：%s", exc)
            flash("文件已保存，但自动分析失败，请检查文件内容后重试。", "error")
    else:
        flash(
            f"导入完成：成功 0 篇，跳过 {len(skipped)} 篇。",
            "error",
        )

    if skipped:
        flash(skipped_message(skipped), "info")
    return redirect(url_for("index"))


@app.post("/delete/<article_id>")
def delete_article(article_id: str):
    """删除指定文献，并同步更新文件、索引和报告状态。"""
    entries = load_index()
    target = next(
        (item for item in entries if str(item.get("id")) == article_id), None
    )
    if target is None:
        flash("没有找到要删除的文献，可能已被删除。", "error")
        return redirect(url_for("index"))

    filename = str(target.get("filename", ""))
    path = safe_library_path(filename)
    if path is None:
        flash("文献文件名不合法，已取消删除。", "error")
        return redirect(url_for("index"))

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        app.logger.warning("删除文献 %s 失败：%s", path, exc)
        flash("删除文件失败，请稍后重试。", "error")
        return redirect(url_for("index"))

    remaining = [item for item in entries if str(item.get("id")) != article_id]
    try:
        refresh_library_index(remaining)
        REPORT_FILE.unlink(missing_ok=True)
        flash(f"已删除《{target.get('title', filename)}》。", "success")
    except Exception as exc:
        app.logger.exception("删除后更新索引失败：%s", exc)
        flash("文件已删除，但索引更新失败，请刷新页面确认。", "error")

    return redirect(url_for("index"))


@app.post("/generate")
def generate():
    """用 library/ 中的全部文献重新生成 report.html。"""
    entries = load_index()
    if not entries:
        flash("文献库还是空的，请先上传 .txt 文件。", "error")
        return redirect(url_for("index"))

    try:
        articles = analyzer.load_articles(LIBRARY_DIR)
        text_tool.generate_report(articles, LIBRARY_DIR, REPORT_FILE)
        save_index(sync_index(entries, articles))
    except (FileNotFoundError, ValueError) as exc:
        flash(f"生成失败：{exc}", "error")
        return redirect(url_for("index"))
    except Exception as exc:
        app.logger.exception("生成报告失败：%s", exc)
        flash("生成报告失败，请检查文献文件后重试。", "error")
        return redirect(url_for("index"))

    flash(f"已用 {len(articles)} 篇文献生成分析报告。", "success")
    return redirect(url_for("view_report"))


@app.get("/report")
def view_report():
    """返回由文献库生成的 report.html。"""
    if not load_index() or not REPORT_FILE.exists():
        flash("报告还没有生成，请先点击“生成分析报告”。", "info")
        return redirect(url_for("index"))

    response = send_file(REPORT_FILE, mimetype="text/html")
    # 重新生成后立即看到新内容，避免浏览器使用旧缓存。
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    """上传文件超过限制时返回首页提示。"""
    flash("上传内容超过 80 MB，请分批导入。", "error")
    return redirect(url_for("index"))


@app.after_request
def disable_html_cache(response):
    """页面 HTML 不缓存，避免浏览器继续使用旧版上传控件。"""
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


ensure_storage()


if __name__ == "__main__":
    print("文献库已启动：http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务。")
    app.run(host="127.0.0.1", port=5000, debug=False)
