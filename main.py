from single_shot_detector import SingleShotDetector

if __name__ == '__main__':
    ssd = SingleShotDetector()
    ssd.detection("./img_test.jpg")
