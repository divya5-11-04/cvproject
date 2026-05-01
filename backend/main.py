import os
import pickle
import json
import re
from typing import Optional, List
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="SmartFace Attendance API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"
KNOWN_FACES_PATH = DATASET_PATH / "known_faces"
ENCODINGS_DIR = BASE_DIR / "encodings"
ENCODINGS_FILE = ENCODINGS_DIR / "encodings.pickle"
STUDENTS_META_FILE = ENCODINGS_DIR / "students_meta.json"
LOGS_PATH = BASE_DIR / "attendance_logs"
PROCESSED_PATH = LOGS_PATH / "processed"
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

KNOWN_FACES_PATH.mkdir(parents=True, exist_ok=True)
ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Helper function to load encodings
def load_encodings():
    if ENCODINGS_FILE.exists():
        with ENCODINGS_FILE.open("rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "encodings" in data and "names" in data:
                return data
    return {"encodings": [], "names": []}


def load_student_meta():
    if STUDENTS_META_FILE.exists():
        with STUDENTS_META_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    return {}


def save_student_meta():
    with STUDENTS_META_FILE.open("w", encoding="utf-8") as f:
        json.dump(students_meta, f, indent=2)


def save_encodings():
    with ENCODINGS_FILE.open("wb") as f:
        pickle.dump({"encodings": known_data["encodings"], "names": known_data["names"]}, f)


def normalize_name(raw_name: str) -> str:
    cleaned = " ".join(raw_name.strip().split())
    return cleaned


def normalize_text(raw: str) -> str:
    return " ".join((raw or "").strip().split())


def upsert_student_meta(name: str, class_name: str = "", group_name: str = ""):
    students_meta[name] = {
        "group_name": normalize_text(group_name),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_student_meta()


def get_student_meta(name: str):
    return students_meta.get(name, {"group_name": "Unassigned"})


def append_face_sample(name: str, face_encoding, face_crop_bgr, suffix: str = "sample", group_name: str = ""):
    safe_name = normalize_name(name)
    if not safe_name:
        raise ValueError("Student name cannot be empty.")

    student_dir = KNOWN_FACES_PATH / safe_name
    student_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = student_dir / f"{suffix}_{timestamp}.jpg"
    cv2.imwrite(str(file_path), face_crop_bgr)

    known_data["encodings"].append(face_encoding)
    known_data["names"].append(safe_name)
    upsert_student_meta(safe_name, group_name=group_name)
    save_encodings()
    refresh_search_index()


def get_largest_face_location(face_locations):
    if not face_locations:
        return None
    return max(
        face_locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
    )


def open_camera_device():
    camera_options = [
        lambda: cv2.VideoCapture(0, cv2.CAP_DSHOW),
        lambda: cv2.VideoCapture(0),
        lambda: cv2.VideoCapture(1, cv2.CAP_DSHOW),
        lambda: cv2.VideoCapture(1),
    ]

    for opener in camera_options:
        cam = opener()
        if cam is not None and cam.isOpened():
            return cam
        if cam is not None:
            cam.release()
    return None

known_data = load_encodings()
students_meta = load_student_meta()
last_auto_update = {}
known_matrix = np.empty((0, 128), dtype=np.float32)
group_to_indexes = {}
camera_stop_requested = False


def refresh_search_index():
    global known_matrix, group_to_indexes
    if len(known_data["encodings"]) == 0:
        known_matrix = np.empty((0, 128), dtype=np.float32)
        group_to_indexes = {}
        return

    known_matrix = np.asarray(known_data["encodings"], dtype=np.float32)
    group_to_indexes = {}
    for idx, name in enumerate(known_data["names"]):
        group_name = normalize_text(get_student_meta(name).get("group_name", ""))
        if not group_name:
            continue
        group_to_indexes.setdefault(group_name.lower(), []).append(idx)


def ensure_metadata_coverage():
    changed = False
    for name in set(known_data["names"]):
        if name not in students_meta:
            students_meta[name] = {
                "group_name": "Unassigned",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            changed = True
    if changed:
        save_student_meta()


ensure_metadata_coverage()
refresh_search_index()


def match_face_encoding(face_encoding, group_name: str = ""):
    if known_matrix.shape[0] == 0:
        return "Unknown", 1.0

    query = np.asarray(face_encoding, dtype=np.float32)
    group_key = normalize_text(group_name).lower()

    if group_key and group_key in group_to_indexes:
        subset_indexes = np.asarray(group_to_indexes[group_key], dtype=np.int32)
        subset_matrix = known_matrix[subset_indexes]
        distances = np.linalg.norm(subset_matrix - query, axis=1)
        local_best_idx = int(np.argmin(distances))
        best_distance = float(distances[local_best_idx])
        global_idx = int(subset_indexes[local_best_idx])
    else:
        distances = np.linalg.norm(known_matrix - query, axis=1)
        global_idx = int(np.argmin(distances))
        best_distance = float(distances[global_idx])

    # Face recognition default tolerance is around 0.6
    if best_distance <= 0.6:
        return known_data["names"][global_idx], best_distance
    return "Unknown", best_distance

def mark_attendance(name, group_name: str = ""):
    """
    Mark attendance into today's CSV file.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    log_file = LOGS_PATH / f"Attendance_{date_str}.csv"
    
    group_name = normalize_text(group_name)

    # Simple check if file exists to write headers
    df = None
    if log_file.exists():
        df = pd.read_csv(log_file)
        if "Group" not in df.columns:
            df["Group"] = ""

        # Check if user already marked today for same group
        already_marked = (
            (df["Name"] == name)
            & (df["Group"].fillna("") == group_name)
        ).any()
        if already_marked:
            return False # Already marked
    else:
        df = pd.DataFrame(columns=['Name', 'Group', 'Time'])
        
    # Append new record
    new_record = pd.DataFrame([{'Name': name, 'Group': group_name, 'Time': time_str}])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(log_file, index=False)
    return True


def sanitize_filename(name: str) -> str:
    if not name:
        return "Unassigned"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())


def update_group_attendance_excel(group_name: str, present_names: List[str], timestamp: Optional[datetime] = None):
    """
    Update or create an Excel workbook per group. First column is `Name` (student names).
    Each new attendance mark adds a new column with date-time label and fills 'Present'/'Absent'.
    """
    ts = timestamp or datetime.now()
    col_label = ts.strftime("%Y-%m-%d %H:%M:%S")
    safe_group = sanitize_filename(group_name or "Unassigned")
    excel_file = LOGS_PATH / f"Attendance_{safe_group}.xlsx"

    # Build list of students for this group from known_data and students_meta
    group_key = normalize_text(group_name).lower()
    group_students = sorted({
        name
        for name in set(known_data["names"])
        if normalize_text(get_student_meta(name).get("group_name", "")).lower() == group_key
    })
    if not group_students:
        # fallback to present names
        group_students = sorted(set(present_names))

    # Read existing workbook if present
    if excel_file.exists():
        try:
            df = pd.read_excel(excel_file, dtype=str)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            df = pd.DataFrame([{"Name": s} for s in group_students])
        else:
            if "Name" not in df.columns:
                # ensure Name column exists
                df.insert(0, "Name", df.index.astype(str))

            # Ensure all group students are present as rows
            existing = df["Name"].astype(str).tolist()
            for s in group_students:
                if s not in existing:
                    df = pd.concat([df, pd.DataFrame([{"Name": s}])], ignore_index=True)
    else:
        df = pd.DataFrame([{"Name": s} for s in group_students])

    # Add or overwrite the timestamp column with Present/Absent
    df[col_label] = df["Name"].apply(lambda n: "Present" if n in present_names else "Absent")

    # Save workbook (pandas will use openpyxl if available)
    try:
        df.to_excel(excel_file, index=False)
    except Exception:
        # Last-resort: write CSV if Excel write fails
        csv_file = LOGS_PATH / f"Attendance_{safe_group}.csv"
        df.to_csv(csv_file, index=False)


class GroupAttendancePayload(BaseModel):
    group_name: str
    present: List[str] = []
    timestamp: Optional[str] = None


@app.post("/group_attendance")
def group_attendance(payload: GroupAttendancePayload):
    """Accepts a JSON body with `group_name`, `present` list and optional `timestamp` (ISO string).
    Updates the group's Excel sheet and also marks individual CSV attendance entries.
    """
    ts = None
    if payload.timestamp:
        try:
            ts = datetime.fromisoformat(payload.timestamp)
        except Exception:
            ts = datetime.now()

    update_group_attendance_excel(payload.group_name, payload.present or [], timestamp=ts)

    # Also mark individual attendance into daily CSVs for compatibility
    for name in payload.present or []:
        try:
            mark_attendance(name, group_name=payload.group_name)
        except Exception:
            continue

    return {"status": "ok", "marked": len(payload.present or [])}


def maybe_auto_update_student(name, face_encoding, face_crop_bgr, confidence_distance):
    now = datetime.now()
    last_time = last_auto_update.get(name)
    if confidence_distance > 0.38:
        return
    if last_time and (now - last_time).total_seconds() < 300:
        return

    meta = get_student_meta(name)
    append_face_sample(
        name,
        face_encoding,
        face_crop_bgr,
        suffix="auto",
        group_name=meta.get("group_name", ""),
    )
    last_auto_update[name] = now

# --- API ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with FRONTEND_INDEX.open("r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "known_faces": len(known_data["names"]),
        "students": len(set(known_data["names"])),
        "groups": len({normalize_text(v.get("group_name", "")) for v in students_meta.values() if normalize_text(v.get("group_name", ""))}),
    }


@app.get("/groups")
def list_groups():
    groups = {}
    for _, meta in students_meta.items():
        group_name = normalize_text(meta.get("group_name", ""))
        if not group_name:
            continue
        groups[group_name] = groups.get(group_name, 0) + 1

    return {
        "groups": [{"group_name": group_name, "count": count} for group_name, count in sorted(groups.items())]
    }


@app.get("/students")
def list_students(group_name: str = ""):
    group_name = normalize_text(group_name)

    students = []
    for name in sorted(set(known_data["names"])):
        meta = get_student_meta(name)
        student_group = normalize_text(meta.get("group_name", ""))
        if group_name and student_group.lower() != group_name.lower():
            continue
        students.append({"name": name, "group_name": student_group})

    return {"students": students}


@app.get("/attendance/today")
def attendance_today(group_name: str = ""):
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_PATH / f"Attendance_{date_str}.csv"
    if not log_file.exists():
        return {"records": []}

    df = pd.read_csv(log_file)
    if "Group" not in df.columns:
        df["Group"] = ""

    group_name = normalize_text(group_name)

    if group_name:
        df = df[df["Group"].fillna("").str.lower() == group_name.lower()]

    records = df.to_dict(orient="records")
    return {"records": records}


@app.get("/processed/{filename}")
def get_processed_image(filename: str):
    target = PROCESSED_PATH / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(target)

def generate_frames(group_name: str = ""):
    """Generator function that yields video frames from webcam."""
    global known_data
    global camera_stop_requested
    camera_stop_requested = False
    camera = open_camera_device()
    if camera is None:
        return
    try:
        while True:
            if camera_stop_requested:
                break
            success, frame = camera.read()
            if not success:
                break

            try:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert BGR to RGB and ensure contiguous memory for dlib
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

                # Find all the faces and face encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except Exception:
                continue
            
            face_names = []
            face_dist_values = []
            for face_encoding in face_encodings:
                name, best_distance = match_face_encoding(face_encoding, group_name=group_name)
                if name != "Unknown":
                    meta = get_student_meta(name)
                    mark_attendance(name, group_name=meta.get("group_name", ""))
                
                face_names.append(name)
                face_dist_values.append(best_distance)
                
            # Display results on the main frame (not the small one)
            for (top, right, bottom, left), name, face_encoding, best_distance in zip(face_locations, face_names, face_encodings, face_dist_values):
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Auto-update dataset for confident recognized faces
                if name != "Unknown":
                    top_clip = max(top, 0)
                    left_clip = max(left, 0)
                    bottom_clip = min(bottom, frame.shape[0])
                    right_clip = min(right, frame.shape[1])
                    face_crop = frame[top_clip:bottom_clip, left_clip:right_clip]
                    if face_crop.size > 0:
                        maybe_auto_update_student(name, face_encoding, face_crop, best_distance)
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()
        camera_stop_requested = False

@app.get("/video_feed")
def video_feed(group_name: str = ""):
    return StreamingResponse(generate_frames(group_name=group_name), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/stop_camera")
def stop_camera():
    global camera_stop_requested
    camera_stop_requested = True
    return {"message": "Camera stop requested"}

@app.post("/register")
async def register_student(
    name: str = Form(...),
    group_name: str = Form(""),
    file: UploadFile = File(...),
):
    """
    Registers a new student by saving their photo and updating encodings.
    """
    name = normalize_name(name)
    if not name:
        raise HTTPException(status_code=400, detail="Student name is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is required")

    np_arr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb)
    if len(face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face found in the image")

    largest_face = get_largest_face_location(face_locations)
    top, right, bottom, left = largest_face
    face_encodings = face_recognition.face_encodings(image_rgb, known_face_locations=[largest_face])
    if len(face_encodings) == 0:
        raise HTTPException(status_code=400, detail="Face encoding failed")

    face_crop = image_bgr[max(top, 0):bottom, max(left, 0):right]
    if face_crop.size == 0:
        raise HTTPException(status_code=400, detail="Failed to crop face")

    append_face_sample(name, face_encodings[0], face_crop, suffix="manual", group_name=group_name)
    return {"message": f"Successfully registered {name}"}


@app.post("/register_capture")
async def register_student_by_capture(
    name: str = Form(...),
    group_name: str = Form(""),
    shots: int = Form(8),
):
    name = normalize_name(name)
    if not name:
        raise HTTPException(status_code=400, detail="Student name is required")

    target_shots = max(7, min(8, int(shots)))
    camera = open_camera_device()
    if camera is None:
        raise HTTPException(status_code=500, detail="Webcam not available")

    captured = 0
    attempts = 0
    max_attempts = 220
    frame_skip = 6

    try:
        while captured < target_shots and attempts < max_attempts:
            ret, frame = camera.read()
            if not ret:
                attempts += 1
                continue

            attempts += 1
            if attempts % frame_skip != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            face_locations = face_recognition.face_locations(rgb_frame)
            largest_face = get_largest_face_location(face_locations)
            if largest_face is None:
                continue

            top, right, bottom, left = largest_face
            encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=[largest_face])
            if len(encodings) == 0:
                continue

            top_clip = max(top, 0)
            left_clip = max(left, 0)
            bottom_clip = min(bottom, frame.shape[0])
            right_clip = min(right, frame.shape[1])
            face_crop = frame[top_clip:bottom_clip, left_clip:right_clip]
            if face_crop.size == 0:
                continue

            append_face_sample(name, encodings[0], face_crop, suffix="capture", group_name=group_name)
            captured += 1
    finally:
        camera.release()

    if captured < 7:
        raise HTTPException(
            status_code=400,
            detail=f"Captured only {captured} face samples. Ensure face is visible and try again.",
        )

    return {"message": f"Successfully registered {name} with {captured} photos."}


@app.post("/group_attendance")
async def group_attendance(file: UploadFile = File(...), group_name: str = Form("")):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is required")

    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized = []
    unknown_count = 0

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name, best_distance = match_face_encoding(face_encoding, group_name=group_name)
        if name != "Unknown":
            meta = get_student_meta(name)
            mark_attendance(name, group_name=meta.get("group_name", ""))
            recognized.append(name)

            face_crop = frame[max(top, 0):bottom, max(left, 0):right]
            if face_crop.size > 0:
                maybe_auto_update_student(name, face_encoding, face_crop, best_distance)
        else:
            unknown_count += 1

        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    out_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    out_path = PROCESSED_PATH / out_name
    cv2.imwrite(str(out_path), frame)

    return {
        "message": "Group attendance processed",
        "recognized": sorted(list(set(recognized))),
        "unknown_count": unknown_count,
        "processed_image_url": f"/processed/{out_name}",
    }


@app.post("/recognize_frame")
async def recognize_frame(file: UploadFile = File(...), group_name: str = Form("")):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is required")

    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized = []
    unknown_count = 0
    detections = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name, best_distance = match_face_encoding(face_encoding, group_name=group_name)
        if name != "Unknown":
            meta = get_student_meta(name)
            mark_attendance(name, group_name=meta.get("group_name", ""))
            recognized.append(name)

            face_crop = frame[max(top, 0):bottom, max(left, 0):right]
            if face_crop.size > 0:
                maybe_auto_update_student(name, face_encoding, face_crop, best_distance)
        else:
            unknown_count += 1

        detections.append(
            {
                "name": name,
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "left": int(left),
                "distance": float(best_distance),
            }
        )

    return {
        "recognized": sorted(list(set(recognized))),
        "unknown_count": unknown_count,
        "detections": detections,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
