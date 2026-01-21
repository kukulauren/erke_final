from pipeline import Prediction
from app.variables import MODEL_PATH, VIDEO_PATH
from flask import Flask, request, jsonify
import threading

model=Prediction(MODEL_PATH, VIDEO_PATH)

app = Flask(__name__)

@app.route("/start_prediction", methods=["POST"])
def start_prediction():
    global prediction_thread
    try:
        if model.running:
            return jsonify({"message": "Prediction already running"}), 400

        prediction_thread = threading.Thread(target=model.start_prediction)
        model.thread = prediction_thread
        prediction_thread.start()
        return jsonify({"message": "Prediction started"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop_prediction", methods=["POST"])
def stop_prediction():
    try:
        data = request.get_json(force=True)
        pos_member = data.get("pos_member")
        pos_wallet = data.get("pos_wallet")
        voucher_number = data.get("voucher_number")
        cashier_id = data.get("cashier_id")

        if not isinstance(pos_member, bool) or not isinstance(pos_wallet, bool):
            return jsonify({"error": "pos_member and pos_wallet must be boolean"}), 400

        if not voucher_number or not cashier_id:
            return jsonify({"error": "voucher_number and cashier_id are required"}), 400
        output,developer_message=model.print_output(pos_wallet, pos_member)
        video_saved = model.stop_prediction(path=rf"E:\IGS_record\{voucher_number}_{cashier_id}")
        return jsonify({
            "prediction_summary": output,
            "recording_saved": video_saved
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

