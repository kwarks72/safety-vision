import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env 파일의 키를 읽어 옴

DOC_DIR = Path("docs/regulations")
INDEX_DIR = Path("rag_index")
EMBED_MODEL = "text-embedding-3-small"  # OpenAI 임베딩 모델
MAX_CHARS = 450  # 한 조각의 최대 글자 수
BATCH = 64  # 한 번에 보낼 조각 수

_client = None
_index = None

# "제32조(보호구의 지급 등)" 처럼 조문이 시작하는 줄을 찾는 규칙
ARTICLE_RE = re.compile(r"(?m)^(?=제\d+조(?:의\d+)?\()")
TITLE_RE = re.compile(r"^(제\d+조(?:의\d+)?\([^)]*\))")


def get_client():
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(".env 파일에 OPENAI_API_KEY가 없습니다.")
        _client = OpenAI()
    return _client


def embed(texts):
    """글을 의미 좌표(숫자 목록)로 바꿉니다."""
    vectors = []
    for i in range(0, len(texts), BATCH):
        resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts[i : i + BATCH])
        vectors.extend(d.embedding for d in resp.data)
        print(f"  좌표 변환 {min(i + BATCH, len(texts))}/{len(texts)}")
    arr = np.array(vectors, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def read_document(path):
    """txt 또는 pdf 파일에서 글자를 읽어 옵니다."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # 쪽마다 반복되는 머리글('법제처 … 국가법령정보센터')은 검색에 방해라 지웁니다
        lines = [
            ln
            for ln in text.split("\n")
            if "국가법령정보센터" not in ln and ln.strip() != "법제처"
        ]
        return "\n".join(lines)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def split_long(text):
    """긴 글을 MAX_CHARS 안팎으로 줄 단위로 나눕니다."""
    parts, buf = [], ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > MAX_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += line + "\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


def make_chunks():
    """문서를 조문 단위로 나누고, 모든 조각 앞에 '조문 제목'을 붙입니다."""
    chunks = []
    for path in sorted(DOC_DIR.glob("*")):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        text = read_document(path)
        for article in ARTICLE_RE.split(text):
            article = article.strip()
            if not article:
                continue
            m = TITLE_RE.match(article)
            title = m.group(1) if m else ""
            for i, part in enumerate(split_long(article)):
                if len(part) < 20:
                    continue
                if title and i > 0:  # 첫 조각은 이미 제목으로 시작함
                    part = f"[{title}] {part}"
                chunks.append({"source": path.name, "text": part})
    return chunks


def build_index():
    """모든 조각을 좌표로 바꿔 저장합니다. (한 번만 실행)"""
    chunks = make_chunks()
    if not chunks:
        raise SystemExit("docs/regulations 폴더에 .txt 또는 .pdf 파일이 없습니다.")
    print(f"조각 {len(chunks)}개를 만들었습니다. 좌표로 바꾸는 중...")
    vectors = embed([c["text"] for c in chunks])
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print("색인 저장 완료:", INDEX_DIR)


def load_index():
    global _index
    if _index is None:
        vectors = np.load(INDEX_DIR / "vectors.npy")
        chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        _index = (vectors, chunks)
    return _index


ARTICLE_ID_RE = re.compile(r"^\[?(제\d+조(?:의\d+)?)")
MAX_PER_ARTICLE = 2  # 한 조문이 후보를 독차지하지 못하게 하는 상한

# (질문에 이런 말이 있으면, 이런 말이 든 조각에 가산점)
KEYWORD_GROUPS = [
    (["안전모", "헬멧"], ["안전모"]),
    (["안전대", "추락", "높은 곳", "고소"], ["안전대"]),
    (["처벌", "벌금", "징역", "과태료", "벌칙"], ["벌칙", "징역", "벌금", "과태료"]),
]


def search(query, top_k=10):
    """질문과 의미가 가까운 조각을 찾습니다.
    - 법령 파일마다 골고루 뽑고, 한 조문이 후보를 독차지하지 못하게 합니다.
    - 질문에 핵심 단어(안전모, 처벌 등)가 있으면 그 단어가 든 조각에 가산점을 줍니다."""
    vectors, chunks = load_index()
    q = embed([query])[0]
    scores = vectors @ q  # 좌표가 가까울수록 점수가 높음

    for triggers, boosts in KEYWORD_GROUPS:
        if any(t in query for t in triggers):
            for i, c in enumerate(chunks):
                if any(b in c["text"] for b in boosts):
                    scores[i] += 0.05

    def article_of(i):
        m = ARTICLE_ID_RE.match(chunks[i]["text"])
        return (chunks[i]["source"], m.group(1)) if m else None

    picked, used = [], {}

    def try_add(i):
        a = article_of(i)
        if a is not None and used.get(a, 0) >= MAX_PER_ARTICLE:
            return False
        picked.append(i)
        if a is not None:
            used[a] = used.get(a, 0) + 1
        return True

    sources = sorted({c["source"] for c in chunks})
    per_source = max(top_k // len(sources), 1)
    for src in sources:
        order = sorted(
            (i for i, c in enumerate(chunks) if c["source"] == src),
            key=lambda i: -scores[i],
        )
        n = 0
        for i in order:
            if n >= per_source:
                break
            if try_add(i):
                n += 1
    for i in np.argsort(-scores):  # 남는 자리는 전체 점수 순으로 채움
        if len(picked) >= top_k:
            break
        i = int(i)
        if i not in picked:
            try_add(i)

    # 보호구·착용 + 처벌 질문이면 '근로자 의무(제40조)'와 '과태료(제175조)' 조각을 꼭 포함
    penalty = any(w in query for w in ["처벌", "벌금", "징역", "과태료", "벌칙"])
    ppe = any(w in query for w in ["안전모", "헬멧", "보호구", "안전대", "착용", "미착용"])
    if penalty and ppe:
        for i, c in enumerate(chunks):
            t = c["text"]
            if "산업안전보건법" not in c["source"]:
                continue
            is40 = t.startswith("제40조(")
            is175 = t.startswith("[제175조") and "제32조제1항" in t and "제40조" in t
            if (is40 or is175) and i not in picked:
                picked.append(i)
                scores[i] = max(scores[i], 0.5)

    picked.sort(key=lambda i: -scores[i])
    return [{"score": float(scores[i]), **chunks[i]} for i in picked]


if __name__ == "__main__":
    build_index()