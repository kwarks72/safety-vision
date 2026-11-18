from collections import Counter

from rag import make_chunks

chunks = make_chunks()
print("전체 조각:", len(chunks))
for src, n in Counter(c["source"] for c in chunks).items():
    print(f"  {n}개 | {src}")

print("\n'안전모'가 들어간 조각:")
for c in chunks:
    if "안전모" in c["text"]:
        print("  -", c["source"][:10], "|", c["text"][:80].replace("\n", " "))