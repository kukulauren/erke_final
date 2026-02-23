from pipeline import Prediction
from app.variables import MODEL_PATH, VIDEO_PATH
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
        # Change fps to 25 when requested
        model.set_target_fps(25)
        output_dir = r"E:\IGS_record"  # same disk as final storage
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
        
        # Disable recording for this sale
        model.disable_recording()
        time.sleep(0.3)
        # Get prediction output with error handling
        try:
            output, developer_message = model.print_output(pos_wallet, pos_member)
        except Exception as e:
            print(f"Error generating output summary: {e}")
            output = {"error": "Failed to generate prediction output"}
            developer_message = {"error": str(e)}
        
        # Save video only if suspicious activity detected
        video_saved = False
        recording_debug = {
            "suspicious": model.suspicious,
            "temp_path": model.temp_video_path,
            "temp_exists": False,
            "output_dir_exists": False,
            "e_drive_exists": os.path.exists("E:\\"),
            "cwd": os.getcwd(),
            "running_user": getpass.getuser(),
            "error": None
        }

        output_dir = r"E:\IGS_record"
        recording_debug["output_dir_exists"] = os.path.exists(output_dir)

        if model.temp_video_path:
            recording_debug["temp_exists"] = os.path.exists(model.temp_video_path)

        if model.suspicious:

            if not model.temp_video_path:
                recording_debug["error"] = "temp_video_path is None"

            elif not os.path.exists(model.temp_video_path):
                recording_debug["error"] = "temp video file does not exist"

            else:
                try:
                    os.makedirs(output_dir, exist_ok=True)

                    safe_voucher = re.sub(r'[\\/:*?"<>|]', "_", voucher_number)
                    output_path = os.path.join(output_dir, f"{safe_voucher}.mp4")

                    os.replace(model.temp_video_path, output_path)
                    video_saved = True

                except PermissionError as e:
                    recording_debug["error"] = f"PermissionError: {str(e)}"

                except Exception as e:
                    recording_debug["error"] = f"Unexpected error: {str(e)}"

        else:
            # Not suspicious → delete temp file
            if model.temp_video_path and os.path.exists(model.temp_video_path):
                try:
                    os.remove(model.temp_video_path)
                except Exception as e:
                    recording_debug["error"] = f"Cleanup error: {str(e)}"
        # Reset for next sale (thread-safe reset)
        model.temp_video_path = None
        model.suspicious = False
        # Reset analytics for next transaction
        with model._lock:
            model.analytics = RetailAnalytics()
        
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
