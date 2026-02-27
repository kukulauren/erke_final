## Configuration

The easy-to-edit configuration values live in `app/variables.py`:

- `VIDEO_PATH` — input video file path or RTSP stream URL.
- `OUTPUT_DIR` - output disk name and directory name


## API Documentation

python version 3.9+

Run the following commands.

```bash
python.exe -m pip install --upgrade pip
python -m venv venv
venv\Scripts\activate
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
  "voucher_number": "<voucher-id>"
}
```

Example using curl:

```bash
curl -X POST http://localhost:8000/stop_prediction -H "Content-Type: application/json" -d "{\"pos_member\": true, \"pos_wallet\": false, \"voucher_number\": \"ABC123\"}"
```

Response (200) example:

Suspicious case
```json
{"developer_message":{"customer_detection":"POSM1-MODELC0","member_detection":"POSM1-MODELM0"},"prediction_summary":{"cashier":true,"customer_paid_cash":true,"customer_paid_wallet":false,"items_scanned":true,"member_use":false,"pos_member":true,"purchasing_customer":false,"scanner_moving":true,"suspicious_activity":true},"recording_debug":{"cwd":"C:\\Users\\user1\\Desktop\\erke_final-main\\erke_final-main","error":null,"output_dir_exists":true,"running_user":"user1","suspicious":true,"temp_exists":true,"temp_path":"D:\\IGS_record\\txn_1772212031108.mp4"},"recording_saved":true}
```

Passed case
```json
{"developer_message":{"customer_detection":"POSM1-MODELC0","member_detection":"POSM1-MODELM0"},"prediction_summary":{"cashier":true,"customer_paid_cash":true,"customer_paid_wallet":false,"items_scanned":true,"member_use":false,"pos_member":true,"purchasing_customer":false,"scanner_moving":true,"suspicious_activity":true},"recording_debug":{"cwd":"C:\\Users\\user1\\Desktop\\erke_final-main\\erke_final-main","error":null,"output_dir_exists":true,"running_user":"user1","suspicious":true,"temp_exists":true,"temp_path":"D:\\IGS_record\\txn_1772212031108.mp4"},"recording_saved":true}
```

Notes about the API: `/start_prediction` starts a background thread and returns immediately; if prediction is already running it returns an error. `/stop_prediction` reads the provided JSON and calls into the `Prediction` object's `print_output` and `stop_prediction` methods, returning the summary and whether a suspicious recording was saved.
