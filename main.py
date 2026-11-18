import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import ask as ask_chatbot
from detector import analyze

app = FastAPI(title="safety-vision")

DB_PATH = "safety.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FRAMES = 60  # 한 영상에서 검사할 최대 장면 수

GUIDE_QUESTION = (
    "작업자가 안전모를 착용하지 않은 것이 감지되었습니다. "
    "관련 규정과 사업주·근로자의 의무를 알려 주세요."
)
_guide_cache = {"answer": None}  # 위반 안내 답변은 한 번만 만들어 기억해 둡니다


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                helmet INTEGER NOT NULL,
                violations INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_image TEXT NOT NULL
            )
            """
        )


init_db()


def save_record(filename, r, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO records (created_at, filename, helmet, violations, status, result_image) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, filename, r["helmet"], r["violations"], status, r["result_image"]),
        )
        return cur.lastrowid, now


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    """사진을 받아 분석하고, 결과를 장부에 기록합니다."""
    ext = Path(file.filename).suffix or ".jpg"
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    r = analyze(str(saved_path), save_dir=str(UPLOAD_DIR))
    status = "violation" if r["violations"] > 0 else "ok"
    record_id, now = save_record(file.filename, r, status)

    return {
        "id": record_id,
        "created_at": now,
        "helmet": r["helmet"],
        "violations": r["violations"],
        "status": status,
    }


@app.post("/analyze-video")
def analyze_video(file: UploadFile = File(...)):
    """영상을 1초에 한 장씩 검사하고, 위반 장면만 기록합니다."""
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(round(fps)), 1)

    results = []
    index = 0
    while len(results) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frame_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(frame_path), frame)
            r = analyze(str(frame_path), save_dir=str(UPLOAD_DIR))
            results.append((int(index / fps), r))
        index += 1
    cap.release()

    if not results:
        raise HTTPException(status_code=400, detail="영상을 읽을 수 없습니다.")

    violation_items = [(sec, r) for sec, r in results if r["violations"] > 0]
    to_save = violation_items if violation_items else results[:1]
    for sec, r in to_save:
        status = "violation" if r["violations"] > 0 else "ok"
        save_record(f"{file.filename} ({sec}초)", r, status)

    return {
        "frames_checked": len(results),
        "violation_frames": len(violation_items),
        "max_violations": max((r["violations"] for _, r in results), default=0),
        "status": "violation" if violation_items else "ok",
    }


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """산업안전 규정 챗봇: 질문 → 관련 조문 검색 → AI 답변"""
    question = req.question.strip()[:500]
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    try:
        return {"answer": ask_chatbot(question)["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 오류: {e}")


@app.get("/guide")
def violation_guide():
    """위반이 감지됐을 때 보여 줄 관련 규정 안내 (한 번만 만들고 기억)"""
    if _guide_cache["answer"] is None:
        try:
            _guide_cache["answer"] = ask_chatbot(GUIDE_QUESTION)["answer"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"안내 생성 오류: {e}")
    return {"answer": _guide_cache["answer"]}


@app.get("/records")
def list_records(limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/stats")
def stats():
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_cases,
                COALESCE(SUM(helmet), 0) AS helmet,
                COALESCE(SUM(violations), 0) AS violations,
                COALESCE(SUM(CASE WHEN status = 'violation' THEN 1 ELSE 0 END), 0) AS violation_cases
            FROM records
            """
        ).fetchone()
    people = row["helmet"] + row["violations"]
    rate = round(row["violations"] / people * 100, 1) if people else 0
    return {
        "total_cases": row["total_cases"],
        "helmet": row["helmet"],
        "violations": row["violations"],
        "violation_cases": row["violation_cases"],
        "violation_rate": rate,
    }


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")