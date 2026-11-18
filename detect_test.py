from ultralytics import YOLO

# 1. 안전모 모델 불러오기 (AI의 눈)
model = YOLO("models/best.pt")
# 모델이 구분할 수 있는 이름표 확인
print("모델이 아는 이름표:", model.names)

# 2. 사진 한 장 분석하기
results = model("samples/test.jpg")

# 3. 무엇을 찾았는지 글자로 출력하기
for box in results[0].boxes:
    name = model.names[int(box.cls)]
    conf = float(box.conf)
    print(f"발견: {name} (확신도 {conf:.0%})")

# 4. 박스가 그려진 결과 사진 저장하기
results[0].save("samples/result.jpg")
print("결과 사진 저장 완료: samples/result.jpg")