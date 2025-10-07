"""
modeling.py

Contains functions for training and evaluating machine learning models
to predict laptop prices.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_model(df, target_col="Price"):
    """
    Train a linear regression model on the laptop dataset.

    Parameters:
        df (pd.DataFrame): Dataset with features and target.
        target_col (str): Name of the target column.

    Returns:
        model (LinearRegression): Trained regression model.
        X_test, y_test: Test set for evaluation.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Prints RMSE and R² metrics.
    """
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

