#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文文章分析核心模块。

本模块只负责“读文件 + 分析”，不涉及 HTML 渲染和 Web 路由：
1. 读取 .txt 文章，第一行作为标题；
2. jieba 分词并过滤停用词；
3. 使用 TF-IDF 提取每篇文章的 Top5 关键词；
4. 使用 TF-IDF 对句子打分，抽取每篇文章分数最高的 2 句核心句。

命令行工具 text_tool.py 与网页应用 app.py 共用这里的函数。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 关闭 jieba 首次加载词典时的提示，保持控制台输出干净。
jieba.setLogLevel(logging.ERROR)

# 中文和英文常见停用词。停用词不参与关键词、核心句的 TF-IDF 计算。
STOPWORDS = set(
    """
    的 了 和 是 在 也 就 都 而 及 与 着 或 一个 我们 你们 他们 她们 它们
    这 那 这些 那些 这个 那个 有 对 从 被 到 为 以 于 中 上 下 等 更
    并 但 又 很 将 让 其 该 已 还 可 会 要 把 给 没有 通过 进行 以及
    一个 一家 一所 一些 不少 许多 相关 有关 方面 问题 情况 工作 目前
    已经 近日 本周 今年 去年 同时 不过 下一步 进一步 根据 计划 主要
    其中 记者 报道 表示 认为 介绍 指出 提醒 据悉 据了解 可以 能够 需要
    这种 这样 那么 什么 怎么 为什么 如果 因为 所以 虽然 但是 而且 并且
    对于 关于 由于 随着 不断 正在 仍然 还是 只是 非常 比较 更加 目前
    the and for with from this that are was were will has have not but
    """.split()
)

# 句子切分：按中文/英文句末标点和换行切分，保留原标点。
SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


@dataclass
class Article:
    """一篇文章的原始内容与分析结果。"""

    path: Path
    title: str
    body: str
    char_count: int
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    core_sentences: List[str] = field(default_factory=list)


def read_text_file(path: Path) -> str:
    """读取文本，优先 UTF-8，兼容 Windows 上常见的 GB18030 编码。"""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"[提示] {path.name} 不是 UTF-8 编码，已按 GB18030 读取。")
        return path.read_text(encoding="gb18030")


def parse_article(path: Path) -> Article | None:
    """解析一篇文章：第一行是标题，其余内容为正文。"""
    raw = read_text_file(path).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        print(f"[跳过] {path.name} 是空文件。")
        return None

    lines = raw.split("\n")
    title_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if title_index is None:
        print(f"[跳过] {path.name} 没有有效内容。")
        return None

    title = lines[title_index].strip()
    body = "\n".join(lines[title_index + 1 :]).strip()
    if not body:
        print(f"[跳过] {path.name} 只有标题，没有正文。")
        return None

    return Article(
        path=path,
        title=title,
        body=body,
        char_count=count_characters(body),
    )


def count_characters(text: str) -> int:
    """统计字数：不计空白字符，保留中文、字母、数字和标点。"""
    return len(re.sub(r"\s+", "", text))


def tokenize(text: str) -> List[str]:
    """jieba 分词，并过滤空白、标点、单字中文和停用词。"""
    tokens: List[str] = []
    for word in jieba.cut(text):
        word = word.strip().lower()
        if not word or word in STOPWORDS:
            continue
        # 单个汉字通常信息量低；英文缩写、型号（如 AI、5G）保留。
        if len(word) == 1 and not word.isascii():
            continue
        # 必须至少包含一个中文、字母或数字，过滤纯标点。
        if not TOKEN_PATTERN.search(word):
            continue
        tokens.append(word)
    return tokens


def make_vectorizer() -> TfidfVectorizer:
    """创建统一的 TF-IDF 向量化器。"""
    return TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        token_pattern=None,
        sublinear_tf=True,
        norm="l2",
        smooth_idf=True,
    )


def extract_keywords(
    texts: Sequence[str], top_n: int = 5
) -> List[List[Tuple[str, float]]]:
    """在整个语料上计算 TF-IDF，返回每篇文章权重最高的 top_n 个词。"""
    if not texts:
        return []

    # 极端情况下所有文本都没有有效词，直接返回空结果。
    if not any(tokenize(text) for text in texts):
        return [[] for _ in texts]

    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    results: List[List[Tuple[str, float]]] = []
    for row in matrix:
        pairs = [
            (feature_names[col], float(row[0, col])) for col in row.indices
        ]
        # 权重降序；权重相同时按词语拼音顺序稳定输出。
        pairs.sort(key=lambda item: (-item[1], item[0]))
        results.append(pairs[:top_n])
    return results


def split_sentences(body: str) -> List[str]:
    """按句末标点和换行切分句子，返回不改写的原句。"""
    sentences: List[str] = []
    for match in SENTENCE_PATTERN.finditer(body):
        sentence = match.group(0).strip()
        # 过滤过短片段（如单独一个小标题或标点）。
        if len(re.sub(r"\s+", "", sentence)) >= 6:
            sentences.append(sentence)
    return sentences


def extract_core_sentences(
    sentence_lists: Sequence[Sequence[str]], top_n: int = 2
) -> List[List[str]]:
    """对全语料句子做 TF-IDF 打分，抽取每篇文章分数最高的 top_n 句。

    打分使用句子中所有词的 TF-IDF 权重之和。选出的句子按原文顺序返回，
    方便读者顺着文章逻辑阅读，且句子内容完全保持原样。
    """
    # 建立“文章下标 -> 句子列表”的扁平索引。
    entries: List[Tuple[int, int, str]] = []
    for article_index, sentences in enumerate(sentence_lists):
        for sentence_index, sentence in enumerate(sentences):
            if tokenize(sentence):
                entries.append((article_index, sentence_index, sentence))

    if not entries:
        # 没有可打分句子时，退化为每篇前两句。
        return [list(sentences[:top_n]) for sentences in sentence_lists]

    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform([item[2] for item in entries])

    scored: dict[int, List[Tuple[float, int]]] = {}
    for row_index, (article_index, sentence_index, _) in enumerate(entries):
        score = float(matrix[row_index].sum())
        scored.setdefault(article_index, []).append((score, sentence_index))

    results: List[List[str]] = []
    for article_index, sentences in enumerate(sentence_lists):
        candidates = scored.get(article_index, [])
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen_indexes = sorted(index for _, index in candidates[:top_n])
        if not chosen_indexes:
            chosen_indexes = list(range(min(top_n, len(sentences))))
        results.append([sentences[index] for index in chosen_indexes])
    return results


def analyze_articles(articles: Sequence[Article]) -> List[Article]:
    """为一组已解析文章补齐关键词、句子和核心句。

    分析仍是“全语料 TF-IDF”，因此命令行和网页共用时结果一致。
    这里只填充传入 Article 对象的分析字段，不改变解析和打分规则。
    """
    article_list = list(articles)
    if not article_list:
        return []

    # 关键词：标题和正文共同参与 TF-IDF。
    keyword_results = extract_keywords(
        [f"{article.title}\n{article.body}" for article in article_list], top_n=5
    )
    for article, keywords in zip(article_list, keyword_results):
        article.keywords = keywords
        article.sentences = split_sentences(article.body)

    core_results = extract_core_sentences(
        [article.sentences for article in article_list], top_n=2
    )
    for article, core_sentences in zip(article_list, core_results):
        article.core_sentences = core_sentences

    return article_list


def load_articles(folder: Path) -> List[Article]:
    """读取文件夹下所有 .txt，并按文件名排序、完成分析。"""
    files = sorted(folder.glob("*.txt"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"文件夹中没有 .txt 文件：{folder}")

    articles: List[Article] = []
    for path in files:
        article = parse_article(path)
        if article:
            articles.append(article)

    if not articles:
        raise ValueError("没有读取到有效文章，请检查文件是否包含标题和正文。")

    return analyze_articles(articles)


def filter_articles(
    articles: Sequence[Article],
    include_terms: Sequence[str],
    exclude_terms: Sequence[str],
) -> List[Article]:
    """按关键词筛选文章。

    include：必须同时包含所有 include 词；
    exclude：只要包含任意一个 exclude 词就排除。
    """
    filtered: List[Article] = []
    for article in articles:
        haystack = f"{article.title}\n{article.body}".casefold()
        if include_terms and not all(
            term.casefold() in haystack for term in include_terms
        ):
            continue
        if any(term.casefold() in haystack for term in exclude_terms):
            continue
        filtered.append(article)
    return filtered
