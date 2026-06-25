from pipeline import Prediction
from app.variables import MODEL_PATH, VIDEO_PATH, OUTPUT_DIR
from app.retail_analytics import RetailAnalytics
from flask import Flask, request, jsonify
import os
import re
import getpass
import time

app = Flask(__name__)

model = None


def initialize_model():
    """Initialize the model and auto-start prediction at 10 fps"""
    global model
    try:
        model = Prediction(MODEL_PATH, VIDEO_PATH, target_fps=10)
        print("✓ Model loaded successfully at startup (target fps: 10)")

        # Auto-start prediction on startup
        model.start_prediction()
        print("✓ Prediction started automatically on startup")
    except Exception as e:
        print(f"✗ Error loading model at startup: {e}")
        raise


@app.before_request
def initialize_once():
    if not hasattr(app, "_model_initialized"):
        initialize_model()
        app._model_initialized = True


@app.route("/start_prediction", methods=["POST"])
def start_prediction():
    try:
        # Change fps to 25 when a sale starts
        model.set_target_fps(25)

        output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        if not model.enable_recording(output_dir=output_dir):
            return jsonify({"error": "Failed to enable recording"}), 500

        return jsonify({"message": "Recording started for sale at 25 fps"}), 200
    except Exception as e:
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

        # ── Step 1: Collect analytics BEFORE stopping the writer ──────────────
        # print_output reads self.analytics which is safe to read while the
        # prediction loop is still running (it only writes analytics state,
        # not the video writer).
        try:
            output, developer_message = model.print_output(pos_wallet, pos_member)
        except Exception as e:
            print(f"Error generating output summary: {e}")
            output = {"error": "Failed to generate prediction output"}
            developer_message = {"error": str(e)}

        # Snapshot suspicious flag NOW before we reset it below
        is_suspicious = model.suspicious
        temp_path_snapshot = model.temp_video_path

        # ── Step 2: Disable recording and wait for the writer to flush ────────
        # wait_for_flush=True ensures the file is completely written before we
        # try to move it.  This is the key fix for the "file not saved" bug.
        model.disable_recording(wait_for_flush=True, timeout=15.0)

        # ── Step 3: Build debug info ──────────────────────────────────────────
        output_dir = OUTPUT_DIR
        recording_debug = {
            "suspicious": is_suspicious,
            "temp_path": temp_path_snapshot,
            "temp_exists": False,
            "output_dir_exists": os.path.exists(output_dir),
            "cwd": os.getcwd(),
            "running_user": getpass.getuser(),
            "error": None
        }

        if temp_path_snapshot:
            recording_debug["temp_exists"] = os.path.exists(temp_path_snapshot)

        # ── Step 4: Move or delete the recording ─────────────────────────────
        video_saved = False

        if is_suspicious:
            if not temp_path_snapshot:
                recording_debug["error"] = "temp_video_path is None"
            elif not os.path.exists(temp_path_snapshot):
                recording_debug["error"] = (
                    "temp video file does not exist after flush – "
                    "recording may have been too short or writer failed to open"
                )
            else:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    safe_voucher = re.sub(r'[\\/:*?"<>|]', "_", voucher_number)
                    output_path = os.path.join(output_dir, f"{safe_voucher}.mp4")
                    os.replace(temp_path_snapshot, output_path)
                    video_saved = True
                    print(f"✓ Suspicious recording saved: {output_path}")
                except PermissionError as e:
                    recording_debug["error"] = f"PermissionError: {str(e)}"
                    print(f"✗ {recording_debug['error']}")
                except Exception as e:
                    recording_debug["error"] = f"Unexpected error: {str(e)}"
                    print(f"✗ {recording_debug['error']}")
        else:
            # Not suspicious → delete temp file
            if temp_path_snapshot and os.path.exists(temp_path_snapshot):
                try:
                    os.remove(temp_path_snapshot)
                    print("✓ Non-suspicious temp recording deleted")
                except Exception as e:
                    recording_debug["error"] = f"Cleanup error: {str(e)}"

        # ── Step 5: Reset state for the next transaction ──────────────────────
        model.temp_video_path = None
        model.suspicious = False

        with model._lock:
            model.analytics = RetailAnalytics()

        # Drop back to monitoring FPS now that the sale is over
        model.set_target_fps(10)

        return jsonify({
            "prediction_summary": output,
            "developer_message": developer_message,
            "recording_saved": video_saved,
            "recording_debug": recording_debug
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
