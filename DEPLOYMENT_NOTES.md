
# Online deployment notes

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - app.py
   - requirements_full.txt renamed to requirements.txt
3. Open Streamlit Community Cloud.
4. Select the repository.
5. Set main file as app.py.
6. Deploy.

Note: `face-recognition` depends on `dlib`, which can fail on some free cloud services.
If it fails, use Render/Railway with Docker or switch to DeepFace/InsightFace.

## Local Windows test

For the full AI version, installation may be easier in Anaconda:

```bash
conda create -n attendance python=3.10
conda activate attendance
pip install -r requirements_full.txt
streamlit run app.py
```

If `dlib` fails on Windows, install Visual Studio Build Tools or use Conda packages.

## Better production architecture

For a real institution:
- Frontend: Streamlit or Django
- Backend: Django/FastAPI
- Database: PostgreSQL
- Storage: encrypted image storage
- Face engine: InsightFace or DeepFace
- Alerts: email/SMS/WhatsApp API
- Hosting: private server or secure cloud
