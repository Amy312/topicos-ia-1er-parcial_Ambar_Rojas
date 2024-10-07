from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    
    segment_polygon = Polygon(segment)
    min_distance = float('inf')
    
    closest_bbox = None
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_polygon = box(x1, y1, x2, y2)
        distance = segment_polygon.distance(bbox_polygon)
        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox

    if min_distance <= max_distance:
        matched_box = closest_bbox
    #print(matched_box)
    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    result = image_array.copy()
    for polygon, box, label in zip(segmentation.polygons, segmentation.boxes, segmentation.labels):
        if label == 'danger':
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        pts = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        colored_mask = np.zeros_like(image_array)
        colored_mask[:] = color

        alpha = 0.5
        mask_bool = mask.astype(bool)
        result[mask_bool] = cv2.addWeighted(
            result[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0
        )
        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return result


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()

        people_indexes = [
            i for i in range(len(labels)) if labels[i] == 0
        ]  
        people_boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in people_indexes
        ]
        people_polygons = [
            [
                [int(coord[0]), int(coord[1])] for coord in results.masks.xy[i]] for i in people_indexes
        ]
        guns = self.detect_guns(image_array, threshold)
        #print(guns, guns.boxes)
        people_labels_txt = []
        for polygon in people_polygons:
            matches = match_gun_bbox(polygon, guns.boxes, max_distance)
            if(matches):
                people_labels_txt.append("danger")
            else:
                people_labels_txt.append("safe")
        
        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(people_polygons),
            polygons=people_polygons,
            boxes=people_boxes,
            labels=people_labels_txt
        )
