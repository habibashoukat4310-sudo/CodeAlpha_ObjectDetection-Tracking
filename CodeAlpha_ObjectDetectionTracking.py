"""Detect and track objects in a webcam, image, or video with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value: str):
    """Treat digit-only sources as camera indexes; leave paths and URLs intact."""
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, image/video path, or stream URL")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights (downloaded on first use)")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--save", type=Path, help="Optional output video path; webcam/video sources only")
    parser.add_argument("--no-display", action="store_true", help="Process frames without opening a window")
    args = parser.parse_args()
    source = parse_source(args.source)
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")

    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
    if args.save and is_image:
        parser.error("--save is for video sources; use --source image.jpg for a displayed still image")

    try:
        model = YOLO(args.model)
    except Exception as exc:
        parser.exit(
            1,
            f"Could not load YOLO model '{args.model}': {exc}\n"
            "Check the weights path or connect to the internet for the first-time download.\n",
        )
    capture = None
    writer = None
    try:
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                raise OSError(f"Could not read image: {source}")
            results = model.track(frame, persist=True, conf=args.confidence, verbose=False)
            annotated = results[0].plot()
            if args.save:
                cv2.imwrite(str(args.save), annotated)
            if not args.no_display:
                cv2.imshow("YOLO Object Detection and Tracking", annotated)
                cv2.waitKey(0)
        else:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise OSError(f"Could not open video source: {source}")
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not create output video: {args.save}")
            last_time = time.perf_counter()
            display_fps = 0.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                results = model.track(frame, persist=True, conf=args.confidence, verbose=False)
                annotated = results[0].plot()
                now = time.perf_counter()
                elapsed = now - last_time
                last_time = now
                if elapsed > 0:
                    display_fps = 0.9 * display_fps + 0.1 / elapsed
                boxes = results[0].boxes
                count = len(boxes) if boxes is not None else 0
                cv2.putText(annotated, f"FPS: {display_fps:.1f}  Objects: {count}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)
                if writer:
                    writer.write(annotated)
                if not args.no_display:
                    cv2.imshow("YOLO Object Detection and Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        if args.save:
            print(f"Saved annotated output to {args.save.resolve()}")
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Object detection failed: {exc}\n")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
