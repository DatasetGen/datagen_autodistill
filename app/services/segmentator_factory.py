from ..segmentators.grounded_sam_segmentator import GroundedSamSegmentator
from ..segmentators.grounding_dino_segmentator import GroundingDinoSegmentator
from ..segmentators.segmentation_bbox_adapter import SegmentationToBoundingBoxAdapter


def get_segmentator(model_name: str, labels):
    if model_name.lower() == "grounded_sam":
        return GroundedSamSegmentator(labels)
    elif model_name.lower() == "grounded_dino":
        return SegmentationToBoundingBoxAdapter(GroundedSamSegmentator(labels))
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
