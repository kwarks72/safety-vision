# safety-vision — 안전모 미착용 감지 + 산업안전 규정 챗봇

공사·제조 현장 사진/영상에서 **안전모 미착용(위반)** 을 AI(YOLO)로 찾아내고,
위반이 감지되면 **관련 법령(산업안전보건법·산업안전보건기준에 관한 규칙)을 자동으로 안내**하는 PoC입니다.
현장 담당자는 대시보드에서 결과를 확인하고, 규정 챗봇에게 궁금한 점을 바로 물어볼 수 있습니다.

![대시보드](docs/dashboard.png)

## 주요 기능

| 기능 | 설명 |
|---|---|
| 안전모 감지 | YOLOv8 모델로 `helmet`(착용) / `head`(미착용) 탐지, 신뢰도 0.5 이상만 사용 |
| 사진·영상 분석 | 이미지 1장 또는 영상(1초당 1프레임, 최대 60프레임) 분석 후 결과 기록 |
| 위반 알림 | 새 위반이 기록되면 대시보드에 경고 배너 + 알림음 + 관련 규정 안내 카드 |
| 규정 챗봇 (RAG) | 법령 원문을 검색해 **근거 조문과 함께** 답변. 근거가 없으면 "확인되지 않습니다"라고 답함 |
| 기록·통계 | SQLite에 분석 이력 저장, 대시보드에서 5초마다 갱신 |

## 구조

```
사진/영상 ─▶ FastAPI ─▶ YOLO 감지(detector.py) ─▶ SQLite(safety.db)
                                  │
                                  └─ 위반 시 ─▶ 규정 안내(/guide) ─▶ 대시보드(static/index.html)

법령 PDF ─▶ 조문 단위 분할 ─▶ 임베딩(OpenAI) ─▶ 검색 ─▶ LLM 답변(chatbot.py)
```

- 백엔드: Python, FastAPI, Uvicorn, SQLite, OpenCV
- AI: Ultralytics YOLOv8, OpenAI `text-embedding-3-small`(검색), `gpt-5-mini`(답변)
- 프런트: 순수 HTML/JS (별도 빌드 없음)

## 실행 방법 (Windows 기준)

```powershell
git clone https://github.com/kwarks72/safety-vision.git
cd safety-vision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. **모델 파일**: 아래 모델의 `best.pt`를 받아 `models/best.pt`로 저장합니다.
   - https://huggingface.co/JarvanLee/yolov8-helmet-violation-detection (Apache-2.0)
2. **법령 PDF**: [국가법령정보센터](https://www.law.go.kr)에서 `산업안전보건법`, `산업안전보건기준에 관한 규칙` PDF를 받아 `docs/regulations/`에 넣습니다.
3. **API 키**: 프로젝트 폴더에 `.env` 파일을 만들고 아래처럼 적습니다. (이 파일은 GitHub에 올라가지 않습니다)
   ```
   OPENAI_API_KEY=여기에_본인_키
   ```
4. 검색 색인 만들기(최초 1회) 후 서버 실행:
   ```powershell
   python rag.py
   uvicorn main:app --reload
   ```
5. 브라우저에서 http://127.0.0.1:8000 접속

## RAG 정확도를 높인 과정

처음에는 한국어 문장 임베딩 모델로 검색했는데, 긴 법조문에서는 엉뚱한 조문이 상위에 올라왔습니다.
아래 순서로 개선했습니다.

1. 임베딩을 OpenAI `text-embedding-3-small`로 교체
2. 조문(제N조) 단위로 자르고, 450자를 넘으면 나눠 **조문 제목을 앞에 붙임**
3. PDF 머리글("국가법령정보센터") 제거
4. 법령별로 골고루 후보를 뽑고, 한 조문이 후보를 독차지하지 못하게 제한
5. 질문 속 핵심어(안전모, 안전대, 처벌 등)가 든 조각에 가산점 부여
6. 보호구·처벌 질문에는 근로자 의무(제40조)와 과태료(제175조) 조각을 반드시 포함
7. 답변 프롬프트에 "근거에 없으면 추측하지 말 것"을 명시

그 결과 "안전모 미착용 시 처벌" 질문에 사업주 의무(제32조), 근로자 의무(제40조),
과태료(제175조⑥ 제3호, 300만원 이하)를 근거로 답하고, 근거 없는 형사처벌은 "확인되지 않음"이라고 답합니다.

## 한계와 개선 계획

- 안전모 외 보호구(안전대, 보안경 등)는 감지하지 않습니다.
- 학습된 공개 모델을 그대로 사용했습니다. 실제 현장 데이터로 추가 학습·평가가 필요합니다.
- 규정 검색은 단순 벡터 검색+키워드 가산점입니다. 이후 재순위화(reranker)와 평가 질문 세트를 추가할 계획입니다.
- 챗봇 답변은 참고용이며 법률 자문이 아닙니다.

## 출처

- 감지 모델: JarvanLee/yolov8-helmet-violation-detection (Hugging Face, Apache-2.0)
- 법령: 국가법령정보센터 (산업안전보건법, 산업안전보건기준에 관한 규칙)
- 데모 이미지는 본인이 촬영했거나 사용 허가가 있는 사진만 사용하세요.

## 만든 사람

곽수연 · sooyeon0702@naver.com · https://github.com/kwarks72
