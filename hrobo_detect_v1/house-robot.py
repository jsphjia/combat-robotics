from ultralytics import YOLO
from roboflow import Roboflow
import numpy as np
import supervision as sv
import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# this function processes each frame from a given video and annotates the detected house robot
def process_frame(frame: np.ndarray, model):
    results = model(frame, imgsz = 1280)[0]
    detections = sv.Detections.from_yolov8(results)
    box_annotator = sv.BoxAnnotator(thickness = 4, text_thickness = 4, text_scale = 2)
    frame = box_annotator.annotate(scene=frame, detections=detections)
    return frame

def main():
    # import most up-to-date dataset from roboflow
    rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
    project = rf.workspace("test-5ev0m").project("brick-detection-original-view")
    version = project.version(2)
    dataset = version.download("yolov8")

    # training YOLO model
    model = YOLO("yolov8s.pt")
    model.train(data = "9-29-data/data.yaml", epochs = 100)

    # import video to test house robot detection
    test_vid_path = "" # add downloaded video path later
    vid_info = sv.VideoInfo.from_video_path(test_vid_path)

    sv.process_video(source_path = test_vid_path, target_path = "results.mp4", callback = process_frame)
    play_video("results.mp4")