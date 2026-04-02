# SmartFace Attendance System - About

## What model is used?

This project uses the Python `face_recognition` library, which is built on top of `dlib`.

Core model components used in this system:
- Face detection: `face_recognition.face_locations(...)`
  - In this codebase, the default detector is used (HOG-based detector in dlib for CPU).
- Face encoding (embedding model): `face_recognition.face_encodings(...)`
  - Produces a 128-dimensional facial embedding vector for each detected face.
- Face matching: `face_recognition.compare_faces(...)` and `face_recognition.face_distance(...)`
  - Matching is done using distance between embeddings (smaller distance = more similar face).

In short: this is an embedding-based face recognition pipeline using dlib models through `face_recognition`, not a custom-trained deep-learning model inside this repository.

## How it works (end-to-end)

### Group support
- Every student can now be registered with:
  - `group_name` (for example: `G1`)
- Attendance records now store `Name`, `Group`, and `Time`.
- You can filter attendance by group in the UI and API.
- Group photo attendance can be scoped to a specific group.

### 1) Registration / dataset creation
The system supports two registration methods:
- Upload a student photo.
- Capture 7-8 photos from webcam (`/register_capture`).

For each captured/registered face:
- The largest detected face is selected.
- Its embedding is computed.
- Face crop is stored in `dataset/known_faces/<Student Name>/`.
- Embedding + student name are appended to `encodings/encodings.pickle`.

### 2) Live attendance from webcam
When live scanner is started:
- Frames are read from webcam (`/video_feed`).
- Faces are detected and encoded in each frame.
- Each embedding is compared with known embeddings.
- Best match is selected using minimum face distance.
- If matched, attendance is marked once per day in CSV.
- Camera stream can be explicitly stopped using the Stop button, which calls a backend stop endpoint.

Attendance logs are stored in:
- `attendance_logs/Attendance_YYYY-MM-DD.csv`

### 3) Group photo attendance
When a classroom image is uploaded (`/group_attendance`):
- All faces in the image are detected and encoded.
- Each face is matched against known embeddings.
- Recognized students are marked present.
- An annotated output image is saved to:
  - `attendance_logs/processed/`

### 4) Automatic dataset update
For recognized faces with high confidence (low distance):
- The system may save new face crops automatically to improve variation in dataset over time.
- A cooldown is used to avoid saving too frequently for the same student.

### 5) Large dataset optimization
To improve performance for very large datasets:
- Encodings are cached in a NumPy matrix for vectorized distance computation.
- Group-based indexing is used so matching can run on a smaller subset when group is provided.
- This significantly reduces matching cost when dataset grows across many groups.

## Why this model choice?

This approach is used because it is:
- Fast to build and deploy on CPU.
- Good for small/medium classroom attendance projects.
- Easy to maintain without full model training pipelines.

## Current limitations

- Performance depends on camera quality, lighting, and face angles.
- Similar-looking faces can still create false matches if threshold is too loose.
- No strong anti-spoofing (photo/video attack protection) is implemented yet.
- For very large datasets, a vector database or ANN index is recommended for faster matching.

## Tech stack

- Backend: FastAPI
- Face recognition: face_recognition + dlib
- Vision processing: OpenCV
- Frontend: HTML + Tailwind CSS + JavaScript
- Data storage: local filesystem (`dataset`, `encodings`, `attendance_logs`)
