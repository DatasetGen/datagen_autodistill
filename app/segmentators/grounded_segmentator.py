from abc import ABC, abstractmethod

class GroundedSegmentator(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def build_annotations(self, results, width, height):
        pass
