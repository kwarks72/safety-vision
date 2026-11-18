from pathlib import Path
from ultralytics import YOLO

# AI 모델은 한 번만 불러옵니다 (AI의 눈)
model = YOLO("models/best.pt")


def analyze(image_path, conf=0.5, save_dir="samples"):
    """사진 한 장을 분석해서 안전모 착용/위반 수를 알려 줍니다."""
    results = model(image_path, conf=conf, verbose=False)
    result = results[0]

    counts = {"helmet": 0, "head": 0, "person": 0}
    for box in result.boxes:
        name = model.names[int(box.cls)]
        counts[name] += 1

    # 박스가 그려진 결과 사진 저장
    out_path = Path(save_dir) / f"result_{Path(image_path).name}"
    result.save(str(out_path))

    return {
        "helmet": counts["helmet"],        # 안전모 착용자 수
        "violations": counts["head"],      # 미착용(위반) 수
        "result_image": str(out_path),
    }