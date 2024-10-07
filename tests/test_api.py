import pytest
import numpy as np
from predictor import annotate_detection
from src.models import Detection, PredictionType, Segmentation
from src.predictor import GunDetector, annotate_segmentation, match_gun_bbox
from fastapi.testclient import TestClient
from src.main import app 
import os
client = TestClient(app)

PATH = fr"{os.path.dirname(os.path.abspath(__file__))}\images"

TEST_IMAGE_PATH = fr"{PATH}\test.jpg"

@pytest.fixture(scope="module")
def test_image():
    with open(TEST_IMAGE_PATH, "rb") as f:
        return f.read()

#API TESTS
def test_get_model_info():
    data = {
        "model_name": "Gun detector",
        "gun_detector_model": "DetectionModel",
        "semantic_segmentation_model": "SegmentationModel",
        "input_type": "image",
    }
    response = client.get("/model_info")
    assert response.status_code == 200
    assert response.json() == data
    
    
def test_detect_guns(test_image):

    response = client.post(
        "/detect_guns",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "n_detections" in data
    assert "boxes" in data
    assert "labels" in data
    assert "confidences" in data
    assert data["pred_type"] == "OD"

def test_annotate_guns(test_image):
    response = client.post(
        "/annotate_guns",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"{PATH}\annotated_guns.jpg", "wb") as f:
        f.write(response.content)

def test_detect_people(test_image):
    response = client.post(
        "/detect_people",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "n_detections" in data
    assert "polygons" in data
    assert "boxes" in data
    assert "labels" in data
    assert data["pred_type"] == "SEG"

def test_annotate_people(test_image):
    response = client.post(
        "/annotate_people",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5", "draw_boxes": "true"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"{PATH}\annotated_people.jpg", "wb") as f:
        f.write(response.content)

def test_detect(test_image):
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detection" in data
    assert "segmentation" in data

    detection = data["detection"]
    assert "n_detections" in detection
    assert "boxes" in detection
    assert "labels" in detection
    assert "confidences" in detection
    assert detection["pred_type"] == "OD"

    segmentation = data["segmentation"]
    assert "n_detections" in segmentation
    assert "polygons" in segmentation
    assert "boxes" in segmentation
    assert "labels" in segmentation
    assert segmentation["pred_type"] == "SEG"

def test_annotate(test_image):
    response = client.post(
        "/annotate",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5", "draw_boxes": "true"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"{PATH}\annotated_combined.jpg", "wb") as f:
        f.write(response.content)

def test_guns(test_image):
    response = client.post(
        "/guns",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    guns_list = response.json()
    assert isinstance(guns_list, list)
    for gun in guns_list:
        assert "gun_type" in gun
        assert "location" in gun
        assert gun["gun_type"] in ["pistol", "rifle"]
        assert "x" in gun["location"]
        assert "y" in gun["location"]

def test_people(test_image):
    response = client.post(
        "/people",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    people_list = response.json()
    assert isinstance(people_list, list)
    for person in people_list:
        assert "person_type" in person
        assert "location" in person
        assert "area" in person
        assert person["person_type"] in ["safe", "danger"]
        assert "x" in person["location"]
        assert "y" in person["location"]
        assert isinstance(person["area"], int)


# PREDICTOR TESTS
def test_match_gun_bbox():
    segment = [[0, 0], [0, 10], [10, 10], [10, 0], [20, 10]]
    bboxes = [
        [10, 0, 25, 20],
        [80, 90, 100, 110]
    ]
    max_distance = 15 
    expected_bbox = [10, 0, 25, 20]
    matched_bbox = match_gun_bbox(segment, bboxes, max_distance)
    assert matched_bbox == expected_bbox


def test_annotate_detection():
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)

    detection = Detection(
        pred_type=PredictionType.object_detection,
        n_detections=1,
        boxes=[[25, 25, 75, 75]],
        labels=['pistol'],
        confidences=[0.95]
    )
    annotated_img = annotate_detection(image_array, detection)
    assert not np.array_equal(annotated_img, image_array)

def test_annotate_segmentation():
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    segmentation = Segmentation(
        pred_type=PredictionType.segmentation,
        n_detections=1,
        polygons=[[[30, 30], [30, 70], [70, 70], [70, 30]]],
        boxes=[[30, 30, 70, 70]],
        labels=['safe']
    )
    annotated_img = annotate_segmentation(image_array, segmentation)
    assert not np.array_equal(annotated_img, image_array)


