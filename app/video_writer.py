"""Video recording with H.264 (PyAV/libx264) and OpenCV mp4v fallback.

H.264 gives roughly half the file size of mp4v at the same quality and plays
directly in browsers, so loss-prevention staff can review clips without
downloading a codec. If PyAV is unavailable or the encoder fails to open,
recording transparently falls back to OpenCV's mp4v so a clip is never lost.
"""
import logging

import cv2

from app.variables import H264_CRF, H264_PRESET, VIDEO_CODEC

logger = logging.getLogger(__name__)

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


class _AvH264Writer:

    def __init__(self, path, fps, size):
        width, height = size
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("h264", rate=max(int(round(fps)), 1))
        # libx264 requires even dimensions for yuv420p
        self.stream.width = width - (width % 2)
        self.stream.height = height - (height % 2)
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(H264_CRF), "preset": H264_PRESET}
        self._even_size = (self.stream.width, self.stream.height)
        self._open = True

    def write(self, frame_bgr):
        if not self._open:
            return
        h, w = self._even_size[1], self._even_size[0]
        if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
            frame_bgr = frame_bgr[:h, :w]
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def release(self):
        if not self._open:
            return
        self._open = False
        try:
            for packet in self.stream.encode():  # flush encoder
                self.container.mux(packet)
        finally:
            self.container.close()

    def isOpened(self):
        return self._open


def create_video_writer(path, fps, size):
    """Return an opened writer (PyAV H.264 preferred) or None on failure.

    The returned object supports write(frame_bgr), release(), isOpened() —
    the same surface as cv2.VideoWriter.
    """
    if VIDEO_CODEC.lower() == "h264" and _AV_AVAILABLE:
        try:
            writer = _AvH264Writer(path, fps, size)
            logger.info("Recording with H.264 (crf=%s, preset=%s)", H264_CRF, H264_PRESET)
            return writer
        except Exception:
            logger.exception("PyAV H.264 writer failed, falling back to mp4v")
    elif VIDEO_CODEC.lower() == "h264":
        logger.warning("PyAV not installed, falling back to mp4v (pip install av)")

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        logger.error("Failed to open any video writer for %s", path)
        return None
    logger.info("Recording with OpenCV mp4v")
    return writer
