## Configuration

The easy-to-edit configuration values live in `app/variables.py`:

- `VIDEO_PATH` — input video file path or RTSP stream URL.
- `OUTPUT_DIR` - output disk name and directory name


## API Documentation

python version 3.9+

Run the following commands.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


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
curl -X POST http://localhost:8000/stop_prediction -H "Content-Type: application/json" -d "{\"pos_member\": true, \"pos_wallet\": false, \"voucher_number\": \"ABC123\"}"
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
