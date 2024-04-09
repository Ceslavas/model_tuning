
# Deep Learning Text Processing Tools

## Overview
This program is designed for optimizing and training neural networks for text data classification, utilizing the Reuters dataset. The program performs the following steps:

1. **Loading and Preparing Data:** Fetches the Reuters dataset, vectorizes the text data, and converts labels to one-hot encoding.
2. **Building the Model:** Creates a deep learning model with dynamic numbers of layers and neurons using Keras and TensorFlow.
3. **Hyperparameter Optimization:** Uses Keras Tuner to adjust hyperparameters such as the number of layers, neurons, and learning rate to improve model performance.
4. **Training and Validation:** Trains the model on training data with early stopping and callbacks to prevent overfitting.
5. **Evaluation and Visualization:** Evaluates the model's performance on test data and visualizes the accuracy history of the model.

The program is developed for exploring and optimizing neural networks in text classification tasks, providing tools to combat overfitting and fine-tune models.

## Requirements
- Python 3.8+ (compatibility with other versions of Python is not guaranteed).
- Libraries:
  ```
  numpy
  tensorflow
  keras
  keras-tuner
  matplotlib
  yaml
  ```
  To install all the libraries at once, run the command: `pip install -r requirements.txt`.

## Installation
1. Make sure Python 3.8+ is installed on your system.
2. Clone this repository or download the project files to your computer:
   ```
   git clone https://github.com/Ceslavas/model_tuning.git "D:\your_folder"
   ```

## Configuration
Before using the text processing tools, you need to set up the `config.yaml` file:
1. Specify the number of words to consider in the dataset and hyperparameters for optimization in the `config.yaml` file.

Example `config.yaml` file:
```yaml
tuner:
    settings:
        max_trials: 40
        directory: 'data'
        project_name: 'reuters_tuning'
        overwrite: True
        early_stopping_patience: 2
    search:
        epochs: 10
        validation_split: 0.2

training:
    num_words: 10000
    learning_rate: 0.01
    batch_size: 1024
    epochs: 10000
    validation_split: 0.2
    callbacks:
      early_stopping:
        patience: 5
```

## Running the Project
To use the text processing tools, follow these steps:
1. Open a command line or terminal.
2. Navigate to the directory where the `src\Model_tuning.py` script is located.
3. Enter the command `python Model_tuning.py`.

## Results
The scripts will process the input data according to the specified configurations, performing model training, hyperparameter optimization, and providing a performance evaluation on the test dataset.

## FAQ
**Q:** Can these tools be used for batch processing of multiple datasets?
**A:** Currently, the scripts are designed for processing a single dataset. Batch processing functionality may be added in the future.

## Contributions
Contributions are welcome! If you have suggestions for improvements or new features, please submit a pull request or create an issue.

## License
This project is distributed under the MIT license. See the LICENSE.txt file for details.
