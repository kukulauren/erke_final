from pipeline import Prediction
from app.variables import MODEL_PATH, VIDEO_PATH
from app.retail_analytics import RetailAnalytics
from flask import Flask, request, jsonify
import os
import re
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
        
        # Get prediction output with error handling
        try:
            output, developer_message = model.print_output(pos_wallet, pos_member)
        except Exception as e:
            print(f"Error generating output summary: {e}")
            output = {"error": "Failed to generate prediction output"}
            developer_message = {"error": str(e)}
        
        # Save video only if suspicious activity detected
        video_saved = False
        if model.suspicious and model.temp_video_path and os.path.exists(model.temp_video_path):
            try:
                output_dir = r"E:\IGS_record"
                os.makedirs(output_dir, exist_ok=True)

                safe_voucher = re.sub(r'[\\/:*?"<>|]', "_", voucher_number)
                output_path = os.path.join(output_dir, f"{safe_voucher}.mp4")

                if os.path.exists(model.temp_video_path):
                    os.replace(model.temp_video_path, output_path)
                    video_saved = True
                    print(f"✓ Suspicious activity recording saved: {output_path}")
                else:
                    print(f"✗ Temp video not found: {model.temp_video_path}")

            except PermissionError as e:
                print(f"✗ File is locked / still being written: {e}")
            except Exception as e:
                print(f"✗ Error saving video: {e}")
        else:
            # Clean up temp file if not saving
            if model.temp_video_path and os.path.exists(model.temp_video_path):
                try:
                    os.remove(model.temp_video_path)
                except Exception as e:
                    print(f"Error removing temp video: {e}")
        
        # Reset for next sale (thread-safe reset)
        model.temp_video_path = None
        model.suspicious = False
        # Reset analytics for next transaction
        with model._lock:
            model.analytics = RetailAnalytics()
        
        return jsonify({
            "prediction_summary": output,
            "developer_message": developer_message,
            "recording_saved": video_saved
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
