

from .models.binoculars.binoculars import Binoculars
from .models.chatgpt_roberta_detector.chatgpt_detector import ChatGPTDetector
from .models.fast_detectgpt.fast_detectgpt import FastDetectGPT
from .models.radar.radar import Radar
from .models.wavedetect.wavedetect_api import WaveDetect_api

class Detector:
    """Shared interface for all detectors"""

    def inference(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list of scores from 0 to 1 with
        0 indicating likely human-written, and 1 indicating likely machine-generated."""
        pass


def get_detector(detector_name: str, ckpt_name=None) -> Detector:
    if detector_name == "chatgpt-roberta":
        return ChatGPTDetector()
    elif detector_name == "fastdetectgpt":
        return FastDetectGPT()
    elif detector_name == "radar":
        return Radar()
    elif detector_name == "binoculars":
        return Binoculars()
    elif detector_name == 'wavedetect':
        return WaveDetect_api()
    else:
        raise ValueError("Invalid detector name")
