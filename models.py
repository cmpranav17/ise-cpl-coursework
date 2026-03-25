
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_models():
    """
    Returns a dict of { model_name: sklearn pipeline }.
    All models are wrapped in a Pipeline with StandardScaler
    so features are normalised before fitting.
    """
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "GaussianProcess": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=RBF() + WhiteKernel(),
                n_restarts_optimizer=3,
                random_state=42,
                normalize_y=True
            ))
        ]),
    }
    return models