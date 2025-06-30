from app.segmentator.grounding_sam_segmentator import GroundingSamSegmentator
from app.segmentator.grounded_dino_segmentator import GroundedDinoSegmentator

from app.segmentators.segmentation_bbox_adapter import SegmentationToBoundingBoxAdapter


def get_segmentator(model_name: str, labels):
    if model_name.lower() == "grounded_sam":
        return GroundingSamSegmentator(labels)
    elif model_name.lower() == "grounded_dino":
        return SegmentationToBoundingBoxAdapter(GroundingSamSegmentator(labels))
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
