
# AI Attendance System - Streamlit Web Prototype

This is a Python-based online attendance system prototype using:

- Streamlit web interface
- SQLite database
- Admin and faculty login
- Student/class database
- Face enrollment by photo
- AI-based attendance marking from camera/uploaded image
- Attendance reports and CSV export
- Dashboard visualizations
- Absentee marking

## Default login

Username: `admin`  
Password: `admin123`

Change this before real deployment.

---

## Option A: Quick UI test without AI recognition

This lets you test login, class creation, dashboard, database, reports, and interface.

```bash
pip install -r requirements_light.txt
streamlit run app.py
```

The AI face recognition module will show as unavailable.

---

## Option B: Full AI test with face recognition

Install:

```bash
pip install -r requirements_full.txt
streamlit run app.py
```

Then:

1. Login as admin.
2. Create a class.
3. Register students with clear face photos.
4. Go to Mark Attendance.
5. Capture photo with webcam or upload a class/student photo.
6. Click "Run AI Recognition and Mark Present".
7. Finalize absentees.
8. Check Reports and Dashboard.

---

## Recommended testing method

For each student, enroll one clear front-facing photo first.

For attendance testing, start with one student photo at a time, then test small group images.

Use good lighting and front-facing images.

---

## Important privacy note

Face images and face encodings are biometric data. For real school deployment, use written consent, secure storage, role-based access, and a clear data deletion policy.

---

## Suggested next upgrades

1. Live webcam streaming using `streamlit-webrtc`.
2. PostgreSQL instead of SQLite for multi-user deployment.
3. Email/SMS/WhatsApp absentee alerts.
4. Separate parent/student portals.
5. Face anti-spoofing or liveness detection.
6. Cloud deployment on Render, Railway, Hugging Face Spaces, or Streamlit Community Cloud.
