import uuid
from app.segmentator.grounded_segmentator import GroundedSegmentator

# Simulación para estructura. Sustituye con la lógica real.
class GroundingDinoSegmentator(GroundedSegmentator):
    def __init__(self, labels):
        self.labels = labels
        self.class_id_to_label = {i: label.id for i, label in enumerate(labels)}
        # Supón que GroundedDINO ya está inicializado aquí
        # self.model = GroundedDINO(...)

    def predict(self, image):
        # results = self.model.predict(image)
        # return results
        pass

    def build_annotations(self, results, width, height):
        annotations = []
        for i, box in enumerate(results.xyxy):
            x1, y1, x2, y2 = box
            normalized_x = x1 / width
            normalized_y = y1 / height
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            label_id = self.class_id_to_label.get(int(results.class_id[i]), self.labels[0].id)
            annotations.append({
                "type": "bounding_box",
                "id": str(uuid.uuid4()),
                "label": label_id,
                "data": {
                    "point": [normalized_x, normalized_y],
                    "width": normalized_width,
                    "height": normalized_height
                }
            })
        return annotations
