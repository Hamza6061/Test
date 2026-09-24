# Test

## Logistic regression with gradient descent

This repository includes a binary logistic regression implementation trained
from scratch with NumPy mini-batch gradient descent. It includes sigmoid
activation, binary cross-entropy loss, L2 regularization, standardization,
stratified splitting, and accuracy, precision, recall, and F1 evaluation.

No machine-learning library is used for the model or optimization.

Run the reproducible demo:

```bash
python logistic_regression.py
```

To use the Pima Indians Diabetes dataset, pass a headerless CSV containing
eight feature columns followed by the binary `Outcome` column:

```bash
python logistic_regression.py --data path/to/diabetes.csv
```
