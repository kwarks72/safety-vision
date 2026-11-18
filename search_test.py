from rag import search

questions = [
    "안전모는 어떤 작업에서 착용해야 하나요?",
    "근로자가 보호구를 착용하지 않으면 어떻게 되나요?",
    "높은 곳에서 떨어질 위험이 있을 때 필요한 보호구는?",
]

for q in questions:
    print("\n질문:", q)
    for r in search(q, top_k=8):
        preview = r["text"].replace("\n", " ")[:90]
        print(f"  [{r['score']:.2f}] {r['source']} | {preview}...")