import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the parameters
batch_size = 32
num_classes = 2  # Update with the number of classes in your dataset
epochs = 10
input_shape = (224, 224, 3)  # Update with the input image dimensions

# Load and preprocess your own dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split your dataset into training and validation
)

train_generator = train_datagen.flow_from_directory(
    'image_classification/img',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'image_classification/img',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    subset='validation'
)

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the model
model.save("image_classifier_model.h5")
