# Sentiment Analysis of Movie Reviews

This project performs sentiment analysis on a dataset of movie reviews. It implements and compares three linear classifiers: Perceptron, Average Perceptron, and Pegasos.

## File Descriptions

- `main.py`: The main script to run the sentiment analysis project. It loads the data, trains the classifiers, tunes hyperparameters, and evaluates the models.
- `project1.py`: Contains the core implementation of the sentiment analysis algorithms, including:
    - `hinge_loss_single`, `hinge_loss_full`: Functions to calculate hinge loss.
    - `perceptron_single_step_update`, `perceptron`: Implementation of the Perceptron algorithm.
    - `average_perceptron`: Implementation of the Average Perceptron algorithm.
    - `pegasos_single_step_update`, `pegasos`: Implementation of the Pegasos algorithm.
    - `classify`: A function to classify data points using a trained model.
    - `classifier_accuracy`: A function to calculate the accuracy of a classifier.
    - `extract_words`, `bag_of_words`, `extract_bow_feature_vectors`: Functions for text processing and feature extraction.
- `utils.py`: Contains utility functions for loading data, plotting results, and tuning hyperparameters.
- `reviews_train.tsv`, `reviews_val.tsv`, `reviews_test.tsv`: The dataset files containing the movie reviews and their sentiment labels.
- `stopwords.txt`: A list of stopwords to be removed from the text during preprocessing.
- `toy_data.tsv`: A small dataset for testing and debugging the algorithms.

## How to Run

1. **Prerequisites:** Make sure you have Python 3 and the following libraries installed:
   - NumPy
   - Matplotlib

2. **Run the main script:**
   ```bash
   python main.py
   ```
   The `main.py` script is divided into sections for different problems. You can uncomment the code for the specific problem you want to run.

## Dependencies

- Python 3
- NumPy
- Matplotlib

## Results

The project evaluates the performance of the Perceptron, Average Perceptron, and Pegasos classifiers on the movie review dataset. The hyperparameters for each model are tuned using the validation set, and the final performance is reported on the test set. The results include:

- **Accuracy:** The classification accuracy of each model on the training, validation, and test sets.
- **Most Explanatory Words:** The words that are most indicative of positive or negative sentiment, as determined by the learned model weights.
