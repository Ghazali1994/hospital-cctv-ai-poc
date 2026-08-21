# Hospital AI CCTV Video Analytics — Working POC

This is a local/on-premise proof-of-concept for:

1. Person detection and tracking
2. Reception waiting-time calculation
3. Stage-to-stage waiting measurement (Entry → Reception → Doctor/Service)
4. Unattended bag/item detection
5. Last-person association for the item
6. Same-camera re-entry tracking
7. CSV event exports for client review

## Windows — recommended

1. Install Python 3.10–3.13.
2. Open this folder in VS Code.
3. Double-click **`setup_and_run.bat`**. It creates `.venv`, installs all required packages, verifies imports, and starts Streamlit.
4. In VS Code, reload the window if the yellow import warnings remain. The included `.vscode/settings.json` points Pylance to `.venv` automatically.
5. Open the Streamlit URL shown in the terminal, normally `http://localhost:8501`.
6. Upload the CCTV clip.
7. Set the Entry / Reception / Doctor zones in the sidebar.
8. Click **Run POC analysis**.

## Manual Windows setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import cv2,numpy,pandas,streamlit,ultralytics; print('OK')"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then select `.venv\Scripts\python.exe` in **VS Code → Python: Select Interpreter** if Pylance still shows missing imports.

## Linux

```bash
chmod +x run_poc.sh
./run_poc.sh
```

## First run

Ultralytics may download the selected YOLO weights the first time the detector starts. For an on-premise production deployment, the model file can be copied into the application directory and the network can be disabled after installation.

## Important POC assumptions

- Tracking IDs are camera-local. Cross-camera identity is NOT claimed as production-ready in this POC.
- The item detector uses common COCO classes: backpack, handbag, suitcase.
- An item becomes an unattended alert when it is sufficiently stationary for the configured duration and the associated person is outside the configured separation distance.
- The “last person” is the closest tracked person associated with the item during the observation window; this is a POC heuristic, not forensic identification.
- No facial recognition is used.
- Video is processed locally by default.

## Suggested hospital demo flow

Person enters → tracked as Person #ID → Reception zone → waiting timer → Doctor zone → waiting time completed.

For the item demo: Person approaches with bag → bag tracked → person walks away → bag remains stationary → unattended alert → last associated Person #ID.

## Production next steps

Before production, validate camera count, RTSP streams, GPU/server sizing, retention, privacy rules, hospital workflow stages, integration APIs, and whether multi-camera re-identification is required.
