import uuid
import numpy as np
from app.segmentator.grounded_segmentator import GroundedSegmentator

class SegmentationToBoundingBoxAdapter(GroundedSegmentator):
    def __init__(self, segmentator: GroundedSegmentator):
        self.segmentator = segmentator
        self.labels = segmentator.labels
        self.class_id_to_label = segmentator.class_id_to_label

    def predict(self, image):
        return self.segmentator.predict(image)

    def build_annotations(self, results, width, height):
        annotations = []

        for i, mask in enumerate(results.mask):
            mask_array = (mask.astype("uint8")) * 255
            ys, xs = np.where(mask_array > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min = xs.min() / width
            y_min = ys.min() / height
            x_max = xs.max() / width
            y_max = ys.max() / height

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            label_id = self.class_id_to_label.get(int(results.class_id[i]), self.labels[0].id)

            annotations.append({
                "type": "bounding_box",
                "id": str(uuid.uuid4()),
                "label": label_id,
                "data": {
                    "point": [x_min, y_min],
                    "width": bbox_width,
                    "height": bbox_height
                }
            })

        return annotations
