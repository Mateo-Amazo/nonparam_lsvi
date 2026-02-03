from sklearn.model_selection import KFold
import numpy as np

from variational.spline_estimation import get_BSpline_decomposition

def regularization_cst_cv(
    log_density,
    multivariate_samples,
    BSpline_list=None,
    log_bounds=(-3, 0),
    order=4,
    Constraint="Concavity",
    knots=None,
    a=None,
    b=None
    ):

    num = log_bounds[1] - log_bounds[0] + 1
    lambdas = np.logspace(log_bounds[0], log_bounds[1], num, base=10)

    N, dimension = multivariate_samples.shape

    n_splits = min(5, N)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    MSE_matrix = np.zeros((len(lambdas), dimension))

    for d in range(dimension):

        for i, lam in enumerate(lambdas):

            cv_errors = []

            for train_index, val_index in kf.split(multivariate_samples):

                sub_train = multivariate_samples[train_index]
                sub_val = multivariate_samples[val_index]

                f_val = log_density(sub_val)

                spline = get_BSpline_decomposition(
                    target_function=log_density,
                    multivariate_samples=sub_train,
                    BSpline_list=BSpline_list,
                    d=d,
                    order=order,
                    Constraint=Constraint,
                    lam=lam,
                    knots=knots,
                    a=a,
                    b=b
                )

                x_val = sub_val[:, d]
                f_val_pred = spline(x_val)

                cv_error = np.mean((f_val - f_val_pred) ** 2)
                cv_errors.append(cv_error)

            MSE_matrix[i, d] = np.mean(cv_errors)

    return lambdas, MSE_matrix


def get_lambda_cv(log_density, multivariate_samples, BSpline_list=None, dimension=None, log_bounds=(-3,0), order=4, Constraint="Concavity", knots=None, a=None, b=None):

    lambdas, MSE_matrix = regularization_cst_cv(log_density=log_density, multivariate_samples=multivariate_samples, BSpline_list=BSpline_list, log_bounds=log_bounds, order=order, Constraint=Constraint, knots=knots, a=a, b=b)
    optimal_lambdas = lambdas[np.argmin(MSE_matrix, axis=0)]
    return optimal_lambdas