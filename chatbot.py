from rag import get_client, search

CHAT_MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """당신은 산업안전 규정 안내 도우미입니다.
반드시 아래 [근거 조문]에 적힌 내용만 사용해서 답하세요.
- 근거에 없는 내용은 추측하지 말고 "제공된 문서에서는 확인되지 않습니다"라고 답하세요.
- 여러 조문이 이어져 있으면 연결해서 설명하세요(예: 근로자의 준수 의무 조문과 과태료 조문).
- 같은 조문이 두 번 나오면 '[시행일: …]' 표시를 확인하고, 아직 시행 전인 내용이면 그렇다고 알려 주세요.
- 쉬운 한국어로 7문장 이내로 답하고, 마지막 줄에 '근거: 제○○조'처럼 사용한 조문을 적으세요.
- 마지막에 "참고용 안내이며 법률 자문이 아닙니다."를 한 줄 덧붙이세요."""


def ask(question, top_k=10):
    """질문 → 관련 조문 검색 → OpenAI가 근거를 보고 답변"""
    hits = search(question, top_k=top_k)
    context = "\n\n".join(
        f"[{i + 1}] ({h['source'][:14]}) {h['text']}" for i, h in enumerate(hits)
    )
    resp = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[근거 조문]\n{context}\n\n[질문]\n{question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            f"{h['text'][:40].replace(chr(10), ' ')}... (유사도 {h['score']:.2f})"
            for h in hits
        ],
    }


if __name__ == "__main__":
    print("산업안전 규정 챗봇 (끝내려면 빈 줄에서 엔터)")
    while True:
        q = input("\n질문> ").strip()
        if not q:
            break
        r = ask(q)
        print("\n" + r["answer"])
        print("\n[검색된 근거 후보]")
        for s in r["sources"]:
            print(" -", s)