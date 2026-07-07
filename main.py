import logging
import os
import threading

from flask import Flask, jsonify, request

from app.transaction_logger import log_transaction
from app.variables import (
    MODEL_PATH,
    MONITORING_FPS,
    OUTPUT_DIR,
    TRANSACTION_FPS,
    VIDEO_PATH,
)
from pipeline import Prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
_init_lock = threading.Lock()


def initialize_model():
    """Load the model and auto-start monitoring at the idle FPS."""
    global model
    model = Prediction(MODEL_PATH, VIDEO_PATH, target_fps=MONITORING_FPS)
    logger.info("Model loaded (monitoring fps: %s)", MONITORING_FPS)

    if model.start_prediction():
        logger.info("Prediction started automatically on startup")
    else:
        logger.error("Could not open video source %r at startup", VIDEO_PATH)


@app.before_request
def initialize_once():
    # Lazy init so it also works under `flask run`; the lock prevents a
    # double-init when two requests arrive at the same time.
    global model
    if model is None:
        with _init_lock:
            if model is None:
                initialize_model()


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "initializing"}), 503
    return jsonify({"status": "ok", **model.status()}), 200


@app.route("/start_prediction", methods=["POST"])
def start_prediction():
    try:
        # Make sure the monitoring loop is alive (it exits when a video file
        # ends or a stream drops permanently), then switch to sale FPS.
        if not model.running and not model.start_prediction():
            return jsonify({"error": "Failed to open video source"}), 500

        model.set_target_fps(TRANSACTION_FPS)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not model.enable_recording(output_dir=OUTPUT_DIR):
            return jsonify({"error": "Failed to enable recording"}), 500

        return jsonify({"message": f"Recording started for sale at {TRANSACTION_FPS} fps"}), 200
    except Exception as e:
        logger.exception("start_prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/stop_prediction", methods=["POST"])
def stop_prediction():
    try:
        data = request.get_json(force=True)
        pos_member = data.get("pos_member")
        pos_wallet = data.get("pos_wallet")
        voucher_number = data.get("voucher_number")

        if not isinstance(pos_member, bool) or not isinstance(pos_wallet, bool):
            return jsonify({"error": "pos_member and pos_wallet must be boolean"}), 400

        if not voucher_number:
            return jsonify({"error": "voucher_number is required"}), 400

        # 1. Collect analytics before touching the writer. This also sets
        #    model.suspicious, which finalize_recording relies on.
        try:
            output, developer_message = model.print_output(pos_wallet, pos_member)
        except Exception as e:
            logger.exception("Error generating output summary")
            output = {"error": "Failed to generate prediction output"}
            developer_message = {"error": str(e)}

        # 2. Log the raw evidence for calibrate.py before analytics reset.
        log_path = log_transaction(
            model.analytics, output, developer_message,
            voucher_number, pos_member, pos_wallet
        )
        if log_path:
            logger.info("Transaction logged: %s", log_path)

        # 3. Stop recording and wait until the file is fully written so it
        #    can be safely moved or deleted.
        model.disable_recording(wait_for_flush=True, timeout=15.0)

        # 4. Save the clip if suspicious, otherwise delete it.
        video_saved, recording_debug = model.finalize_recording(voucher_number, OUTPUT_DIR)

        # 5. Reset for the next transaction and drop back to monitoring FPS.
        model.reset_analytics()
        model.set_target_fps(MONITORING_FPS)

        return jsonify({
            "prediction_summary": output,
            "developer_message": developer_message,
            "recording_saved": video_saved,
            "recording_debug": recording_debug
        }), 200

    except Exception as e:
        logger.exception("stop_prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
