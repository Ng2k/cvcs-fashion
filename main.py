"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from src.single_shot_detector import SingleShotDetector

if __name__ == '__main__':
    ssd = SingleShotDetector()
    ssd.detection("./static/img_test.jpg")
