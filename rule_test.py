from detector import analyze

for path in ["samples/test.jpg", "samples/test_violation.jpg"]:
    r = analyze(path)
    status = "[위반]" if r["violations"] > 0 else "[정상]"
    print(f"{status} {path} -> 착용 {r['helmet']}명, 미착용 {r['violations']}명")
    print(f"        결과 사진: {r['result_image']}")