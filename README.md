# Retail Checkout Loss-Prevention (ERKE)

CCTV-based transaction verification for a retail POS. A YOLO model (`best.pt`)
watches the checkout counter and detects **cashier, customer, scanner, item,
phone, cash, counter**. Analytics on top of the detections decide whether each
member cash transaction actually had a real customer, real cash, and a member
scan — and records a video clip of any transaction flagged as suspicious.

## How it works

- **Tracking**: BoT-SORT with ReID enabled (`app/botsort_reid.yaml`) so track
  IDs survive occlusion — staff/customer classification and action logic
  depend on stable IDs.
- **Pose estimation**: YOLO11-pose extracts the cashier's wrist positions.
  A wrist on an item or on cash is direct evidence of scanning / cash
  handling, far more reliable than bounding-box overlap.
- **Temporal action recognition** (`app/action_recognizer.py`): item scans,
  phone payments and cash handovers are confirmed from *sustained*
  multi-frame evidence with dropout tolerance — a one-frame proximity blip
  no longer fires an event, and a one-frame detector miss no longer cancels
  a payment in progress. A learned clip classifier can be plugged in via
  `ActionRecognizer.set_model()` once labelled clips exist.
- **Frame queue**: the capture thread is decoupled from inference by a
  bounded queue. On RTSP, if inference falls behind, the oldest frames are
  dropped so analysis stays real-time (`frames_dropped` in `/health`).
- **H.264 recording**: clips are encoded with libx264 via PyAV (~⅓ the size
  of mp4v, browser-playable). Falls back to mp4v automatically if PyAV is
  missing.

## Project layout

```
main.py                    Flask control API (start/stop transaction, health)
pipeline.py                Prediction: capture + worker threads, recording, verdict
calibrate.py               Data-driven threshold suggestions from transaction logs
app/variables.py           All configuration (reads .env, sane defaults)
app/helper_functions.py    Detection parsing, analytics dispatch, frame rendering
app/retail_analytics.py    RetailAnalytics: ledgers, staff/customer, counter dwell
app/action_recognizer.py   Temporal action recognition (scan/payment/cash)
app/pose_estimator.py      YOLO11-pose cashier wrist tracking
app/video_writer.py        H.264 (PyAV) recording with mp4v fallback
app/transaction_logger.py  Per-transaction JSON evidence logs
app/botsort_reid.yaml      BoT-SORT + ReID tracker configuration
testing.py                 Offline harness: annotated video + report from a file
testing_rtsp.py            Quick RTSP connectivity check
```

## Configuration

Everything is configured through `.env` (loaded by `app/variables.py`).
Key values:

- `VIDEO_PATH` — input video file path **or** RTSP stream URL
  (e.g. `rtsp://user:pass@192.168.1.10:554/stream1`)
- `MODEL_PATH` — YOLO weights (default `best.pt`)
- `OUTPUT_DIR` — where suspicious transaction clips are saved
- `MONITORING_FPS` / `TRANSACTION_FPS` — processing rate when idle vs. during a sale
- Detection/analytics thresholds — see comments in `.env`

## Setup

Python 3.9+

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the API

```bash
python main.py
```

The API listens on port `8000`. On the first request the model loads and
monitoring starts automatically at `MONITORING_FPS`. If an RTSP stream drops,
the pipeline reconnects automatically.

### `GET /health`

Returns pipeline state: whether the loop is running, frames processed,
recording status, and live analytics counters. Useful for POS integration
checks and watchdogs.

### `POST /start_prediction` — sale started

No body required. Raises processing to `TRANSACTION_FPS` and starts recording
the transaction to a temp file in `OUTPUT_DIR`.

```bash
curl -X POST http://localhost:8000/start_prediction
```

### `POST /stop_prediction` — sale finished

Body:

```json
{
  "pos_member": true,
  "pos_wallet": false,
  "voucher_number": "ABC123"
}
```

```bash
curl -X POST http://localhost:8000/stop_prediction -H "Content-Type: application/json" -d "{\"pos_member\": true, \"pos_wallet\": false, \"voucher_number\": \"ABC123\"}"
```

Decision logic:

- **Wallet payment** (`pos_wallet: true`) — POS already confirmed payment, CV
  is not enforced. Never suspicious.
- **Member cash transaction** (`pos_member: true`) — the CV pipeline must have
  seen a customer at the counter, cash, and a member scan (phone near
  scanner). If any is missing, `suspicious_activity: true` and the clip is
  saved as `<voucher_number>.mp4` in `OUTPUT_DIR`.
- **Non-member cash** — CV is not enforced.

Non-suspicious clips are deleted. The response includes the full
`prediction_summary` (all CV signals are real observations, not hardcoded),
`developer_message` codes for whichever signal was missing, and
`recording_debug` info. After the response, analytics reset for the next
transaction and processing drops back to `MONITORING_FPS`.

Developer message codes: `POSM1-MODELC0` no customer seen, `POSM1-MODELB0`
no cash seen, `POSM1-MODELM0` no member scan seen.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the temporal action recognizer (evidence accumulation, dropout
tolerance, cooldowns, wrist evidence), analytics ledgers, counter dwell,
staff classification, the full suspicious-activity decision matrix, the
H.264 writer + mp4v fallback, transaction logging and calibration math.
No YOLO models are loaded — the suite runs in a few seconds.

## Threshold calibration

Every `/stop_prediction` writes a JSON evidence log to `TRANSACTION_LOG_DIR`
(default `logs/transactions`). Once real transactions have accumulated,
suggest data-driven thresholds from the observed distributions:

```bash
python calibrate.py            # report current vs suggested thresholds
python calibrate.py --write    # apply suggestions with enough samples to .env
```

Suggestions are only trusted at 30+ samples per metric (`--min-samples`).

## Offline testing

Process a recorded video with the exact same analytics code the server runs,
producing an annotated video and a console report:

```bash
python testing.py                       # uses VIDEO_PATH / OUTPUT_PATH from .env
python testing.py testvideo.mp4 out.mp4
```

To verify camera connectivity only:

```bash
python testing_rtsp.py
```
