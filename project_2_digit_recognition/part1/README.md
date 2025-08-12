# Machine Learning on MNIST

This project implements and evaluates several fundamental machine learning classification algorithms on the MNIST handwritten digit dataset. It serves as a practical exercise in understanding models like Linear Regression, Support Vector Machines (SVM), and Softmax Regression, along with feature engineering techniques like Principal Component Analysis (PCA).

## Project Structure

The project is organized into several Python files, each responsible for a specific part of the implementation:

- `main.py`: The main script to run the different models and experiments.
- `utils.py`: Contains utility functions for loading and plotting the MNIST data.
- `linear_regression.py`: Implements a linear regression model with a closed-form solution.
- `svm.py`: Implements both one-vs-rest and multi-class Support Vector Machines.
- `softmax.py`: Implements multinomial logistic (softmax) regression using gradient descent.
- `features.py`: Contains functions for feature engineering, including PCA and cubic feature transformations.
- `kernel.py`: Implements kernel functions for non-linear models.
- `kernel_softmax.ipynb`: A Jupyter Notebook demonstrating a kernelized version of softmax regression.

## Requirements

The project requires the following Python libraries:

- NumPy
- Matplotlib
- scikit-learn (used for SVM implementation)

You can install them using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## How to Run

To run the models and see the results, execute the `main.py` script from the command line:

```bash
python main.py
```

The script will train each of the implemented models, classify the test data, and print the resulting test error for each one.

## Implemented Models and Techniques

This project covers the following algorithms and concepts:

1.  **Linear Regression for Classification**: A linear regression model trained with a closed-form solution to perform classification.
2.  **Support Vector Machine (SVM)**:
    -   **One-vs-Rest SVM**: A binary SVM is trained to distinguish one digit class from the rest.
    -   **Multi-class SVM**: Extends the SVM to handle all 10 digit classes.
3.  **Softmax Regression**: A multinomial logistic regression model trained with batch gradient descent.
4.  **Feature Engineering**:
    -   **Principal Component Analysis (PCA)**: Used for dimensionality reduction to see how the models perform with a smaller feature set.
    -   **Cubic Features**: A non-linear feature transformation is applied to the PCA-reduced data to train a linear model on non-linear features.
5.  **Kernelized Softmax Regression**: The `kernel_softmax.ipynb` notebook explores a more advanced, kernelized version of softmax regression for non-linear classification.
