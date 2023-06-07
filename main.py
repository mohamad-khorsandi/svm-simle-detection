import video_streaming
from smile_detection import SmileDetector


def live_stream():
    sd = SmileDetector('model')
    sd.load_model()
    video_streaming.camera_smile_detection(sd)


def train_and_store():
    sd = SmileDetector('model')
    print('preprocess')
    sd.preprocess()
    print('train')
    sd.train('linear')
    sd.accuracy()
    sd.save_model()


def load_and_test():
    sd = SmileDetector('model')
    sd.load_model()
    print('preprocess')
    sd.preprocess()
    sd.accuracy()


if __name__ == '__main__':
    live_stream()