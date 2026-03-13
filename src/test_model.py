from model import create_model

NUM_CLASSES = 46

model = create_model(NUM_CLASSES)

model.summary()