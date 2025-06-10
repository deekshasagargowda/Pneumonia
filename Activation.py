import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.activations import elu, swish, softmax
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_dir = 'Train'
test_dir = 'Test'

img_width, img_height = 150, 150
batch_size = 32

def load_data(train_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    class_names = list(train_generator.class_indices.keys())
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)

    return train_generator, test_generator, class_names

def train_cnn(train_generator, test_generator, class_names):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=elu, input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation=elu))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation=elu))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation=swish))
    model.add(Dense(len(class_names), activation=softmax))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )

    return model, history

def plot_graphs(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot1.png')
    plt.close()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot1.png')
    plt.close()

def save_model(model, model_name='Pneumonia_classifier2'):
    model.save(f'{model_name}.h5')

def evaluate_model(model, test_generator):
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

    cm = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix1.png')
    plt.close()

if __name__ == "__main__":
    train_generator, test_generator, class_names = load_data(train_dir, test_dir)
    trained_model, training_history = train_cnn(train_generator, test_generator, class_names)
    plot_graphs(training_history)
    save_model(trained_model)
    evaluate_model(trained_model, test_generator)