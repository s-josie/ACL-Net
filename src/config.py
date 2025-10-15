IMAGES_DIR = "data/processed/swimseg/images"
MASKS_DIR = "data/processed/swimseg/GTmaps"
RESIZE_SIZE = (300, 300)
CROP_SIZE = (288, 288)
LOG_DIR = "../logs"
WEIGHTS_DIR = "../weights"
INFERENCE_DIR = "../inference"
NUM_CLASSES = 2
BATCH_SIZE = 2 #batch size of 8 used in paper, otherwise same metrics
INITIAL_LEARNING_RATE = 0.0001
EPOCHS = 300
SEED = 42
TEST_SIZE = 0.2

COLOR_VALUES = {0: [0, 0, 0],
                1: [255, 255, 255]}