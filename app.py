
import os
import sqlite3
import pickle
import hashlib
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px

# Cloud-friendly AI imports: OpenCV-based face detection + image embedding
try:
    import cv2
    FACE_ENGINE_AVAILABLE = True
except Exception:
    FACE_ENGINE_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    BCRYPT_AVAILABLE = False


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
FACE_DIR = DATA_DIR / "student_faces"
DB_PATH = DATA_DIR / "attendance.db"

DATA_DIR.mkdir(exist_ok=True)
FACE_DIR.mkdir(exist_ok=True)


# ---------------------------
# Database setup
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash BLOB NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin', 'faculty')),
        full_name TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        section TEXT,
        UNIQUE(class_name, section)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        class_id INTEGER NOT NULL,
        guardian_contact TEXT,
        email TEXT,
        face_image_path TEXT,
        face_encoding BLOB,
        active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY(class_id) REFERENCES classes(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        class_id INTEGER NOT NULL,
        attendance_date TEXT NOT NULL,
        attendance_time TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('Present', 'Absent', 'Late')),
        confidence REAL,
        marked_by INTEGER,
        source TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(student_id, class_id, attendance_date),
        FOREIGN KEY(student_id) REFERENCES students(id),
        FOREIGN KEY(class_id) REFERENCES classes(id),
        FOREIGN KEY(marked_by) REFERENCES users(id)
    )
    """)

    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        create_user("admin", "admin123", "admin", "System Administrator", conn=conn)

    conn.commit()
    conn.close()


def hash_password(password: str):
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    return hashlib.sha256(password.encode()).hexdigest().encode()


def verify_password(password: str, hashed: bytes):
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password.encode(), hashed)
        except Exception:
            pass
    return hashlib.sha256(password.encode()).hexdigest().encode() == hashed


def create_user(username, password, role, full_name, conn=None):
    own_conn = conn is None
    conn = conn or get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO users(username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, hash_password(password), role, full_name, datetime.now().isoformat(timespec="seconds"))
    )
    if own_conn:
        conn.commit()
        conn.close()


def authenticate(username, password):
    conn = get_conn()
    row = conn.execute("SELECT id, username, password_hash, role, full_name FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row and verify_password(password, row[2]):
        return {"id": row[0], "username": row[1], "role": row[3], "full_name": row[4]}
    return None


# ---------------------------
# Face recognition helpers
# ---------------------------
def pil_to_rgb_array(image: Image.Image):
    return np.array(image.convert("RGB"))


def pil_to_bgr_array(image: Image.Image):
    rgb = pil_to_rgb_array(image)
    if not FACE_ENGINE_AVAILABLE:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_face_crops(image: Image.Image):
    """
    Cloud-friendly face detection using OpenCV Haar cascades.
    Returns aligned/resized grayscale face crops and bounding boxes.
    """
    if not FACE_ENGINE_AVAILABLE:
        return [], [], "AI face engine is not installed. Install requirements.txt and reboot the app."

    bgr = pil_to_bgr_array(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    crops = []
    boxes = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
        crops.append(face)
        boxes.append((int(x), int(y), int(w), int(h)))

    return crops, boxes, f"Detected {len(crops)} face(s)."


def face_embedding_from_crop(face_crop):
    """
    Lightweight face descriptor:
    - resized grayscale face
    - normalized pixel vector
    - L2 normalization

    This is cloud-friendly and works for demonstration/prototype use.
    For production, replace this with InsightFace/DeepFace embeddings.
    """
    vec = face_crop.astype("float32").flatten()
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    norm = np.linalg.norm(vec) + 1e-6
    return vec / norm


def get_face_encoding(image: Image.Image):
    if not FACE_ENGINE_AVAILABLE:
        return None, "AI face engine is not installed. Install requirements.txt and reboot the app."

    crops, boxes, msg = detect_face_crops(image)

    if len(crops) == 0:
        return None, "No face detected. Please use a clear front-facing photo with good lighting."
    if len(crops) > 1:
        return None, "Multiple faces detected. Please upload one student's face only."

    encoding = face_embedding_from_crop(crops[0])
    return encoding, "Face encoded successfully."


def serialize_encoding(encoding):
    return pickle.dumps(encoding)


def deserialize_encoding(blob):
    return pickle.loads(blob)


def save_face_image(student_code, image):
    safe_code = "".join(c for c in student_code if c.isalnum() or c in ("_", "-"))
    path = FACE_DIR / f"{safe_code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image.convert("RGB").save(path)
    return str(path)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6))


def identify_faces(image: Image.Image, class_id: int, tolerance: float = 0.50):
    """
    Matches captured faces against stored student face descriptors.

    The UI tolerance slider is converted to a similarity threshold:
    lower tolerance = stricter, higher tolerance = more permissive.
    """
    if not FACE_ENGINE_AVAILABLE:
        return [], "AI face engine is not installed."

    conn = get_conn()
    rows = conn.execute("""
        SELECT id, student_code, full_name, face_encoding
        FROM students
        WHERE active=1 AND class_id=? AND face_encoding IS NOT NULL
    """, (class_id,)).fetchall()
    conn.close()

    if not rows:
        return [], "No enrolled student faces found for this class. Register students first."

    known_encodings = [deserialize_encoding(r[3]) for r in rows]
    known_meta = [{"student_id": r[0], "student_code": r[1], "full_name": r[2]} for r in rows]

    crops, boxes, detect_msg = detect_face_crops(image)

    if not crops:
        return [], "No face detected in attendance photo."

    # Convert tolerance range to similarity threshold.
    # 0.35 -> very strict around 0.76
    # 0.65 -> more permissive around 0.58
    similarity_threshold = 0.97 - (tolerance * 0.60)

    matches = []
    for crop in crops:
        enc = face_embedding_from_crop(crop)
        sims = [cosine_similarity(enc, known) for known in known_encodings]
        best_idx = int(np.argmax(sims))
        best_similarity = float(sims[best_idx])

        if best_similarity >= similarity_threshold:
            m = dict(known_meta[best_idx])
            m["distance"] = 1.0 - best_similarity
            m["confidence"] = max(0.0, min(1.0, best_similarity))
            matches.append(m)

    unique = {}
    for m in matches:
        if m["student_id"] not in unique or m["confidence"] > unique[m["student_id"]]["confidence"]:
            unique[m["student_id"]] = m

    return list(unique.values()), f"{detect_msg} Matched {len(unique)} student(s)."


# ---------------------------
# Data functions
# ---------------------------
def get_classes():
    conn = get_conn()
    df = pd.read_sql_query("SELECT id, class_name, COALESCE(section, '') AS section FROM classes ORDER BY class_name, section", conn)
    conn.close()
    return df


def add_class(class_name, section):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO classes(class_name, section) VALUES (?, ?)", (class_name, section))
    conn.commit()
    conn.close()


def add_student(student_code, full_name, class_id, guardian_contact, email, face_image, user_id):
    encoding, message = get_face_encoding(face_image)
    if encoding is None:
        return False, message

    img_path = save_face_image(student_code, face_image)

    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO students(student_code, full_name, class_id, guardian_contact, email, face_image_path, face_encoding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_code, full_name, class_id, guardian_contact, email, img_path,
            serialize_encoding(encoding), datetime.now().isoformat(timespec="seconds")
        ))
        conn.commit()
        return True, "Student registered and face enrolled successfully."
    except sqlite3.IntegrityError as e:
        return False, f"Could not register student: {e}"
    finally:
        conn.close()


def get_students(class_id=None):
    conn = get_conn()
    query = """
    SELECT s.id, s.student_code, s.full_name, c.class_name, c.section,
           s.guardian_contact, s.email, s.active
    FROM students s
    JOIN classes c ON s.class_id=c.id
    """
    params = []
    if class_id:
        query += " WHERE s.class_id=?"
        params.append(class_id)
    query += " ORDER BY c.class_name, c.section, s.full_name"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def mark_present(student_id, class_id, confidence, user_id, source="AI face recognition"):
    now = datetime.now()
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO attendance(student_id, class_id, attendance_date, attendance_time, status, confidence, marked_by, source, created_at)
        VALUES (?, ?, ?, ?, 'Present', ?, ?, ?, ?)
    """, (
        student_id, class_id, now.date().isoformat(), now.strftime("%H:%M:%S"),
        confidence, user_id, source, now.isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def mark_absentees(class_id, attendance_date, user_id):
    conn = get_conn()
    students = conn.execute("SELECT id FROM students WHERE class_id=? AND active=1", (class_id,)).fetchall()
    for (sid,) in students:
        exists = conn.execute("""
            SELECT id FROM attendance WHERE student_id=? AND class_id=? AND attendance_date=?
        """, (sid, class_id, attendance_date)).fetchone()
        if not exists:
            conn.execute("""
                INSERT INTO attendance(student_id, class_id, attendance_date, attendance_time, status, confidence, marked_by, source, created_at)
                VALUES (?, ?, ?, ?, 'Absent', NULL, ?, 'System absentee marking', ?)
            """, (
                sid, class_id, attendance_date, datetime.now().strftime("%H:%M:%S"),
                user_id, datetime.now().isoformat(timespec="seconds")
            ))
    conn.commit()
    conn.close()


def attendance_report(start_date, end_date, class_id=None):
    conn = get_conn()
    query = """
    SELECT a.attendance_date, a.attendance_time, s.student_code, s.full_name,
           c.class_name, c.section, a.status, ROUND(a.confidence, 3) AS confidence, a.source
    FROM attendance a
    JOIN students s ON a.student_id=s.id
    JOIN classes c ON a.class_id=c.id
    WHERE a.attendance_date BETWEEN ? AND ?
    """
    params = [start_date, end_date]
    if class_id:
        query += " AND a.class_id=?"
        params.append(class_id)
    query += " ORDER BY a.attendance_date DESC, c.class_name, s.full_name"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


# ---------------------------
# UI helpers
# ---------------------------
def class_selector(label="Select class"):
    classes = get_classes()
    if classes.empty:
        st.warning("No class found. Admin should create a class first.")
        return None, None
    classes["display"] = classes["class_name"] + classes["section"].apply(lambda x: f" - {x}" if x else "")
    selected = st.selectbox(label, classes["display"].tolist())
    class_id = int(classes.loc[classes["display"] == selected, "id"].iloc[0])
    return class_id, selected


def require_login():
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        return st.session_state.user

    st.title("AI Attendance System")
    st.caption("Online Python prototype using Streamlit + SQLite + facial recognition")

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin123")
        submitted = st.form_submit_button("Login")
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.info("Default login: username `admin`, password `admin123`. Change this before real deployment.")
    st.stop()


# ---------------------------
# Pages
# ---------------------------
def dashboard_page(user):
    st.header("Dashboard")

    today = date.today().isoformat()
    df = attendance_report(today, today)

    col1, col2, col3 = st.columns(3)
    col1.metric("Marked today", len(df))
    col2.metric("Present today", int((df["status"] == "Present").sum()) if not df.empty else 0)
    col3.metric("Absent today", int((df["status"] == "Absent").sum()) if not df.empty else 0)

    if not df.empty:
        fig = px.histogram(df, x="status", color="status", title="Today's Attendance Status")
        st.plotly_chart(fig, use_container_width=True)

        trend = attendance_report((date.today().replace(day=1)).isoformat(), today)
        if not trend.empty:
            daily = trend.groupby(["attendance_date", "status"]).size().reset_index(name="count")
            fig2 = px.line(daily, x="attendance_date", y="count", color="status", markers=True, title="Attendance Trend This Month")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No attendance records for today yet.")


def admin_page(user):
    st.header("Admin Panel")

    tabs = st.tabs(["Create class", "Add user", "Register student", "View students"])

    with tabs[0]:
        with st.form("class_form"):
            class_name = st.text_input("Class name", placeholder="Example: Grade 10")
            section = st.text_input("Section", placeholder="Example: A")
            if st.form_submit_button("Save class"):
                if class_name.strip():
                    add_class(class_name.strip(), section.strip())
                    st.success("Class saved.")
                else:
                    st.error("Class name is required.")

    with tabs[1]:
        with st.form("user_form"):
            full_name = st.text_input("Full name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["faculty", "admin"])
            if st.form_submit_button("Create user"):
                if username and password:
                    create_user(username, password, role, full_name)
                    st.success("User created.")
                else:
                    st.error("Username and password are required.")

    with tabs[2]:
        class_id, selected_class = class_selector("Student class")
        with st.form("student_form"):
            student_code = st.text_input("Student ID / Roll number")
            full_name = st.text_input("Student full name")
            guardian_contact = st.text_input("Guardian contact")
            email = st.text_input("Email")
            face_file = st.file_uploader("Upload clear face photo", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Register student")
            if submitted:
                if not all([student_code, full_name, class_id, face_file]):
                    st.error("Student ID, full name, class, and face photo are required.")
                else:
                    image = Image.open(face_file)
                    ok, msg = add_student(student_code.strip(), full_name.strip(), class_id, guardian_contact, email, image, user["id"])
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        if not FACE_ENGINE_AVAILABLE:
            st.warning("OpenCV AI face engine is not installed. Check requirements.txt and reboot the app.")

    with tabs[3]:
        class_filter, class_name = class_selector("Filter by class")
        students = get_students(class_filter)
        st.dataframe(students, use_container_width=True)


def attendance_page(user):
    st.header("Mark Attendance")

    class_id, class_name = class_selector("Select class for attendance")
    if not class_id:
        return

    st.subheader("Capture or upload classroom photo")
    source = st.radio("Image source", ["Camera", "Upload image"], horizontal=True)

    image = None
    if source == "Camera":
        camera_file = st.camera_input("Take a classroom/student photo")
        if camera_file:
            image = Image.open(camera_file)
    else:
        up = st.file_uploader("Upload classroom/student photo", type=["jpg", "jpeg", "png"])
        if up:
            image = Image.open(up)

    tolerance = st.slider("Recognition tolerance", 0.35, 0.65, 0.50, 0.01,
                          help="Lower is stricter; higher may increase false matches.")

    if image:
        st.image(image, caption="Input image", use_container_width=True)
        if st.button("Run AI Recognition and Mark Present"):
            matches, msg = identify_faces(image, class_id, tolerance=tolerance)
            st.info(msg)
            if matches:
                for m in matches:
                    mark_present(m["student_id"], class_id, m["confidence"], user["id"])
                st.success(f"Marked {len(matches)} student(s) present.")
                st.dataframe(pd.DataFrame(matches), use_container_width=True)
            else:
                st.warning("No students matched.")

    st.divider()
    st.subheader("Finalize absentees")
    selected_date = st.date_input("Attendance date", value=date.today())
    if st.button("Mark remaining enrolled students as absent"):
        mark_absentees(class_id, selected_date.isoformat(), user["id"])
        st.success("Absentees marked for selected class/date.")


def reports_page(user):
    st.header("Reports")

    classes = get_classes()
    class_id = None
    if not classes.empty:
        classes["display"] = classes["class_name"] + classes["section"].apply(lambda x: f" - {x}" if x else "")
        options = ["All classes"] + classes["display"].tolist()
        selected = st.selectbox("Class", options)
        if selected != "All classes":
            class_id = int(classes.loc[classes["display"] == selected, "id"].iloc[0])

    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=date.today().replace(day=1))
    end = col2.date_input("End date", value=date.today())

    df = attendance_report(start.isoformat(), end.isoformat(), class_id)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        summary = df.groupby(["class_name", "section", "status"]).size().reset_index(name="count")
        fig = px.bar(summary, x="class_name", y="count", color="status", barmode="group", title="Attendance Summary")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV report", data=csv, file_name="attendance_report.csv", mime="text/csv")

        absentee = df[df["status"] == "Absent"]
        if not absentee.empty:
            st.subheader("Absentee / irregular attendance list")
            st.dataframe(absentee[["attendance_date", "student_code", "full_name", "class_name", "section"]], use_container_width=True)
    else:
        st.info("No records found for the selected period.")


def system_status_page():
    st.header("System Status")
    st.write("Database:", str(DB_PATH))
    st.write("Face engine available:", FACE_ENGINE_AVAILABLE)
    st.write("Face engine:", "OpenCV cloud-friendly detector/descriptor" if FACE_ENGINE_AVAILABLE else "Unavailable")
    st.write("Password hashing:", "bcrypt" if BCRYPT_AVAILABLE else "SHA-256 fallback")
    if not FACE_ENGINE_AVAILABLE:
        st.warning("""
        The app UI and database will work, but AI recognition is disabled.
        To enable it online, make sure requirements.txt contains opencv-python-headless and reboot the app.
        """)


def main():
    st.set_page_config(page_title="AI Attendance System", page_icon="🧑‍🎓", layout="wide")
    init_db()
    user = require_login()

    with st.sidebar:
        st.success(f"Logged in: {user['full_name'] or user['username']} ({user['role']})")
        page_options = ["Dashboard", "Mark Attendance", "Reports", "System Status"]
        if user["role"] == "admin":
            page_options.insert(1, "Admin Panel")
        page = st.radio("Navigation", page_options)
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    if page == "Dashboard":
        dashboard_page(user)
    elif page == "Admin Panel":
        admin_page(user)
    elif page == "Mark Attendance":
        attendance_page(user)
    elif page == "Reports":
        reports_page(user)
    elif page == "System Status":
        system_status_page()


if __name__ == "__main__":
    main()
