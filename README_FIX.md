# Cloud-friendly AI fix for Streamlit

This version removes dlib and face-recognition because they often fail on Streamlit Cloud.

Upload/replace these files in GitHub:

- app.py
- requirements.txt
- packages.txt
- runtime.txt

Then in Streamlit Cloud:

Manage app → Clear cache and reboot

This version uses OpenCV-based face detection and lightweight image matching.
It is suitable for online prototype testing with laptop webcam/photo capture.
For production use, replace the recognition engine with InsightFace/DeepFace and a stronger database backend.
