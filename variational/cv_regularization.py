import numpy as np
from sklearn.model_selection import KFold
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition

def regularization_cst_cv(f, X, num=5, log_bounds=(-3,0), order=4, Constraint="Concavity", a=None, b=None):

    lambdas = np.logspace(log_bounds[0], log_bounds[1], num, base=10)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    MSE_list = []

    for lam in lambdas:
        cv_errors = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            f_val = f(X_val)

            Beta, BSpline_Basis = get_BSpline_decomposition(
                f=lambda x: f(x),
                X=X_train,
                order=order,
                Constraint=Constraint,
                lam=lam,
                a=a,
                b=b
            )

            approx_curve = Curve(BSpline_Basis, Beta)
            f_val_pred = np.array([approx_curve.evaluate(x)[0] for x in X_val])
            cv_error = np.mean((f_val - f_val_pred) ** 2)
            cv_errors.append(cv_error)

        MSE_list.append(np.mean(cv_errors))
    return lambdas, MSE_list