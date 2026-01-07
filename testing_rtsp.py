import cv2
from app.variables import VIDEO_PATH
# Replace this with your RTSP URL
RTSP_URL = VIDEO_PATH

def test_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Failed to connect to RTSP stream.")
        return

    print("Connected to RTSP stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from RTSP stream.")
            break

        # Display the frame
        cv2.imshow("RTSP Stream Test", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("RTSP test finished.")

if __name__ == "__main__":
    test_rtsp_stream(RTSP_URL)