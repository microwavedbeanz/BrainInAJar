import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

title = """
   ___           _        _____         _     __             
  / __\_ __ __ _(_)_ __   \_   \_ __   /_\    \ \  __ _ _ __ 
 /__\// '__/ _` | | '_ \   / /\/ '_ \ //_ \    \ \/ _` | '__|
/ \/  \ | | (_| | | | | /\/ /_ | | | /  _  \/\_/ / (_| | |   
\_____/_|  \__,_|_|_| |_\____/ |_| |_\_/ \_/\___/ \__,_|_|   
                                                                 
"""

# Load the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize the images to be in the range [0, 1]
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define class names for the CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def train():
    # Build the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    epochamount = int(input("How many Epochs would you like to run ?: "))
    print(f"Predicted Time :{(epochamount*10)//60} Minutes and {(epochamount*10)%60} Seconds")
    model.fit(training_images, training_labels, epochs=epochamount, validation_data=(testing_images, testing_labels))

    # Evaluate the model
    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Save the model
    modelname = input("Enter Model Name: ")
    with open("model_data.txt", "a") as f:
        f.writelines(f"\n{modelname}---{round(accuracy,3)}")
    model.save(f'models/{modelname}.keras')
    time.sleep(2)
    menu()


def test():
    #test model
    print("Model Name---Model Accuracy")
    with open("model_data.txt", "r") as f:
        models_list = f.readlines()
        x = 'placeholder'
        i = 0
        while i < len(models_list):
            x = models_list[i]
            print(f'[{i + 1}] {x.strip()}')
            i += 1

        print(f'\n[X] NO MORE MODELS')

        model_select = int(input("Please select Model using reference in [here] :"))
        model_select -= 1
        selected_model = models_list[model_select]
        selected_model = str(selected_model)
        final_model = selected_model.split("---")
        final_model = final_model[0]

            
    model = models.load_model(f"models/{final_model}.keras")

    img = cv.imread('data/lion.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img])/255)
    index = np.argmax(prediction)
    print("-----------------------------------------")
    print(f"Prediction is {class_names[index]}")
    print("-----------------------------------------")
    time.sleep(5)
    menu()
    
def menu():
    print(title)
    print("""
          [1] Train
          [2] Test
          """)
    x = int(input("Enter Choice: "))
    if x == 1:
        train()
    if x == 2:
        test()


menu()