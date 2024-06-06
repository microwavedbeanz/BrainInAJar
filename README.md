# BrainInAJar: A Neural Network Playground

Welcome to **BrainInAJar**, My attempt of a . First Version was made in about an hour following a basic youtube tutorial, i am trying to further my knowledge in the world of machine learning and this seemed like a good step, the version i have uploaded here is much more user friendly, containing a menu and some automation. Id like to add a few more features such as the models.txt file auto-sorting by accuracy. This README.md is mainly so i dont forget how to use it in 6 months
## Getting Started

To start using BrainInAJar, follow these simple steps:

1. **Run the Main Program**
   - Execute `Main.py` from your terminal or IDE of choice.
   - This will launch the menu interface where you can choose to either train a new model or test an existing one.

2. **Choose Your Action**
   - The menu will present two options:
     1. **Train a Model**: Choose this to train a new neural network with your data.
     2. **Test a Model**: Choose this to evaluate the performance of a previously trained model.

3. **Save and Load Models**
   - Trained models are automatically saved to the `models` folder. You can load these models later for testing or further training.

## Important Details

### Image Handling

- **Line 85 in `Main.py`**: Update this line to specify the image you want to process. This ensures the correct image is used for training or testing.
- **Image Requirements**: 
  - All images must be resized to **32x32 pixels**.
  - Add your images to the `data` folder.

### Folder Structure

- **`models` Folder**: This is where your trained models are stored.
- **`data` Folder**: Place all your image data here. Ensure they are properly compressed to 32x32 pixels.

## Enjoy !
