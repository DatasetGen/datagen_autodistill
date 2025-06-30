import cv2
import uuid
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from app.segmentator.grounded_segmentator import GroundedSegmentator

class GroundingSamSegmentator(GroundedSegmentator):
    def __init__(self, labels):
        ontology = {label.prompt: label.id for label in labels}
        self.labels = labels
        self.class_id_to_label = {i: label.id for i, label in enumerate(labels)}
        self.model = GroundedSAM(ontology=CaptionOntology(ontology))

    def predict(self, image):
        return self.model.predict(image)

    def build_annotations(self, results, width, height):
        annotations = []
        for i, mask in enumerate(results.mask):
            mask_uint8 = (mask.astype("uint8")) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            normalized_points = [[pt[0] / width, pt[1] / height] for pt in points]
            xs, ys = zip(*normalized_points)
            left, top = min(xs), min(ys)
            label_id = self.class_id_to_label.get(int(results.class_id[i]), self.labels[0].id)

            annotations.append({
                "id": str(uuid.uuid4()),
                "type": "polygon",
                "is_synthetic": False,
                "data": {
                    "left": left,
                    "top": top,
                    "points": normalized_points
                },
                "label": label_id
            })
        return annotations
