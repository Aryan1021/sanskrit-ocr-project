from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout


def create_model(num_classes):

    model = Sequential()

    # First Convolution Layer
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D((2,2)))

    # Second Convolution Layer
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Third Convolution Layer
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(256, activation='relu'))

    # Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model