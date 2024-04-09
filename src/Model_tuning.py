# Standard library imports
from typing import List, Tuple

# Third-party imports for data manipulation and numerical operations
import numpy as np
import yaml

# TensorFlow and Keras imports for building and training neural network models
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

# Keras Tuner for hyperparameter tuning
import keras_tuner as kt

# Matplotlib for plotting
import matplotlib.pyplot as plt



# New callback for stopping when overfitting
class OverfittingStoppingCallback(Callback):
    def __init__(self, threshold=0.01):
        super(OverfittingStoppingCallback, self).__init__()
        self.threshold = threshold  # The threshold value for the accuracy difference

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get accuracy on training and validation sets
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc is not None and val_acc is not None:
            # Check if the difference in accuracy exceeds the set threshold
            if train_acc - val_acc > self.threshold:
                print(f'\nTraining stopped due to overfitting at epoch {epoch+1}')
                print(f'Accuracy on training set: {train_acc:.4f}, on validation set: {val_acc:.4f}')
                self.model.stop_training = True  # Stop training
                


def load_and_prepare_reuters(num_words: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and prepares the Reuters dataset, including data vectorization and label conversion to one-hot encoding.
    
    Parameters:
    - num_words: The number of most frequent words to consider in the dataset.
    
    Returns:
    - Tuple containing:
      - x_train: Training data vectorized.
      - y_train: Training labels in one-hot encoding.
      - x_test: Test data vectorized.
      - y_test: Test labels in one-hot encoding.
    """
    # Load the Reuters dataset
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)
    # Vectorize training and testing data
    x_train = vectorize_sequences(train_data, num_words)
    x_test = vectorize_sequences(test_data, num_words)
    # Convert labels to one-hot encoding
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return x_train, y_train, x_test, y_test


def vectorize_sequences(sequences: List, dimension: int = 10000) -> np.ndarray:
    """
    Vectorizes sequences, converting them into vectors of 0s and 1s.
    
    Parameters:
    - sequences: A list of sequences to vectorize.
    - dimension: The dimension of the vectorized form.
    
    Returns:
    - A numpy array of vectorized sequences.
    """
    # Create a zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    # Set positions corresponding to the indices of words in the sequences to 1s
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def model_builder(hp: kt.HyperParameters) -> models.Sequential:
    """
    Builds a neural network model using Keras Tuner for hyperparameter optimization.
    
    Arguments:
    - hp: HyperParameters object from Keras Tuner, providing access to the hyperparameter space.
    
    Returns:
    - Compiled Sequential model.
    """
    
    # Creating a Sequential model. The Sequential model is a linear stack of layers.
    model = models.Sequential()
    
    # Adding an input layer with a dynamic number of neurons.
    # The number of neurons is selected from the range [440, 456] with a step of 4.
    # 'relu' activation ensures the non-linearity necessary for learning complex tasks.
    model.add(layers.Dense(
        units=hp.Int('units_in_input_layer', min_value=16, max_value=48, step=8),
        activation='relu',
        input_shape=(10000,),  # Defining the input data shape, necessary for the first layer of the network.
        name=f'input_layer_{hp.Int("unique_id", min_value=0, max_value=1000, step=1)}'  # Unique name for the layer
    ))

    # Dynamically adding hidden layers. The number of these layers can be from 0 to 2.
    # This allows the model to be flexible and adapt to different tasks.
    for i in range(hp.Int('hidden_layers', min_value=0, max_value=2, step=1)):
        model.add(layers.Dense(
            name=f'hidden_layer_{i + 1}',
            units=hp.Int(f'units_in_hidden_layer_{i + 1}', min_value=16, max_value=48, step=8),
            activation='relu'
        ))

    # Adding a Dropout layer to prevent overfitting by "turning off" some neurons during training.
    # The dropout rate is dynamically chosen in the range of 0.1 to 0.5.
    model.add(layers.Dropout(hp.Float('dropout', min_value=0.4, max_value=0.6, step=0.1)))
    
    # Output layer with softmax activation, which returns a probability distribution across 46 classes.
    model.add(layers.Dense(
        name='output_layer',
        units=46,
        activation='softmax'
    ))

    # Compiling the model with a dynamic learning rate.
    # RMSprop optimizer is effective for most types of problems.
    # The 'categorical_crossentropy' loss function is suitable for multi-class classification.
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=hp.Choice('learning_rate', values=[0.01, 0.001])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model  # Returning the model ready for training


def run_tuner(x_train: np.ndarray, y_train: np.ndarray, config: dict) -> models.Sequential:
    """
    Launches the hyperparameter tuning process using Keras Tuner's RandomSearch.
    
    Parameters:
    - x_train: Training data (features).
    - y_train: Training data (labels).
    - config: Configuration parameters.
    
    Returns:
    - The best model found by the tuner.
    """
    
    # Initialize the RandomSearch tuner with configurations from config.yaml
    tuner = kt.RandomSearch(
        hypermodel      = model_builder,  # Model-building function
        objective       = 'val_loss',  # Target metric
        max_trials      = config['tuner']['settings']['max_trials'],  # Number of hyperparameter combinations to try
        directory       = config['tuner']['settings']['directory'],  # Directory to store results
        project_name    = config['tuner']['settings']['project_name'],  # Name of the project
        overwrite       = config['tuner']['settings']['overwrite']  # Overwrite the results if the project already exists
    )
    
    # EarlyStopping callback to stop training if `val_accuracy` does not improve for patience epochs
    overfitting_stopping = OverfittingStoppingCallback(threshold=0.01)
    stop_early = callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience= config['tuner']['settings']['early_stopping_patience']  # Number of epochs with no improvement after which training will be stopped
    )
    
    # Start the search process
    tuner.search(
        x_train, y_train,
        epochs          = config['tuner']['search']['epochs'],
        validation_split= config['tuner']['search']['validation_split'],
        callbacks       = [stop_early, overfitting_stopping]
    )
    
    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Print a summary of the best model
    best_model.summary()

    return best_model


if __name__ == '__main__':
    # Load the configuration file
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
        
    # Extract the configuration parameters
    num_words           = config['training']['num_words']
    learning_rate       = config['training']['learning_rate']
    batch_size          = config['training']['batch_size']
    epochs              = config['training']['epochs']
    validation_split    = config['training']['validation_split']
    patience            = config['training']['callbacks']['early_stopping']['patience']
    
        
    # Load and prepare the Reuters dataset
    x_train, y_train, x_test, y_test = load_and_prepare_reuters(num_words=num_words)
    # Run the tuner to find the best model configuration
    model = run_tuner(x_train, y_train, config)  # Using a part of the test data as a validation set
    model.summary()
    
    min_learning_rate = learning_rate / 1000
    # Utilizing callbacks for training improvements
    # Create an instance of our custom callback
    #overfitting_stopping = OverfittingStoppingCallback(threshold=0.05)
    reuters_callbacks = [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_lr=min_learning_rate),  # Reduces the learning rate if no improvement is seen for 10 epochs
        callbacks.EarlyStopping(monitor='val_accuracy', patience=20),  # Stops training if no improvement is seen for 20 epochs
        callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True),  # Saves the best model
        #overfitting_stopping  # Stops training if overfitting is detected
    ]

    # Train the model with the training set, using a validation split and the specified callbacks
    # Compile the model with the chosen optimizer, loss function, and metrics
    print('\n'*5)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=reuters_callbacks)
    
    # Evaluate the model performance on the test dataset
    test_scores = model.evaluate(x_test, y_test)
    print(test_scores)
    
    # Plotting training history for accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    print()

