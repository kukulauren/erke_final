## API Documentation

## Quick overview

- `main.py` — Flask service exposing two endpoints to control prediction: `/start_prediction` and `/stop_prediction`.
- `pipeline.py` — `Prediction` class that loads a YOLO model, runs tracking loop, performs analytics and optionally saves suspicious video clips.
- `app/helper_functions.py` — Preprocessing, model prediction wrapper, frame rendering and glue logic used by `pipeline.Prediction`.
- `app/retail_analytics.py` — Analytics logic (scanner movement, item scan detection, payments, customer counter detection).
- `app/variables.py` — Project configuration constants (model path, video/RTSP path, thresholds).
- `testing.py` — Standalone reference script showing how video processing + analytics are orchestrated (useful as a reference/experiment).

## Requirements

- Python 3.9+ (project tested with recent 3.x).  Install dependencies with pip.
- The project uses the `ultralytics` YOLO package and `opencv-python` for video I/O and rendering.

Recommended (install into a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python flask
```

If you use a GPU, follow `ultralytics` docs to ensure CUDA/cuDNN and a compatible torch build are installed.

## Configuration

The easy-to-edit configuration values live in `app/variables.py`:

- `MODEL_PATH` — path to the trained weights (default `best.pt`).
- `VIDEO_PATH` — input video file path or RTSP stream URL.
- Thresholds and timing constants (e.g. `CONF_THRESHOLD`, `SCANNER_ITEM_DISTANCE`, `CUSTOMER_DWELL_TIME`, etc.).

Edit those values for your environment (RTSP credentials, local filenames, thresholds).

## How it runs

1. Start the Flask control API:

```bash
python main.py
```

The API listens on port `8000` by default.

2. Start prediction (server-side background thread):

POST to `/start_prediction` (no body required):

Example using curl:

```bash
curl -X POST http://localhost:8000/start_prediction
```

This will create a background thread that runs `Prediction.start_prediction()` which:

- opens the configured `VIDEO_PATH` (file or RTSP stream)
- reads frames, calls the YOLO model, runs analytics and renders frames

3. Stop prediction and request a summary:

POST to `/stop_prediction` with a JSON body containing:

```json
{
  "pos_member": false,
  "pos_wallet": false,
  "voucher_number": "<voucher-id>",
  "cashier_id": "<cashier-id>"
}
```

Example using curl:

```bash
curl -X POST http://localhost:8000/stop_prediction \
  -H "Content-Type: application/json" \
  -d '{"pos_member": false, "pos_wallet": false, "voucher_number": "V123", "cashier_id": "C1"}'
```

Response (200) example:

```json
{
  "prediction_summary": {
    "items_scanned": true,
    "cashier": true,
    "scanner_moving": true,
    "pos_member": false,
    "suspicious_activity": false
  },
  "recording_saved": false
}
```

Notes about the API: `/start_prediction` starts a background thread and returns immediately; if prediction is already running it returns an error. `/stop_prediction` reads the provided JSON and calls into the `Prediction` object's `print_output` and `stop_prediction` methods, returning the summary and whether a suspicious recording was saved.
