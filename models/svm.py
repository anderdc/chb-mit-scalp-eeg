import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC # support vector classification


model = SVC(
    C=1.0,      # Regularization parameter. The strength of the regularization is inversely proportional to C.
    tol=0.01,   # Tolerance for stopping criterion.
    kernel="rbf",
    class_weight="balanced", # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
    random_state=27
    )


