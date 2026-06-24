from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)

# Sliced metrics on categorical groups


def compute_slice_metrics(
    data, categorical_features, model, encoder, lb, output_path="slice_output.txt"
):
    """
    Computes model performance metrics on slices of the data for each categorical feature.
    Writes results to a text file.
    Inputs
    ------
    data : test data from train, test split
    categorical features: list of categorical features
    model: trained model
    encoder:trained encoder
    lb: trained label binarizer
    output_path: path to save output text file

    Returns
    -------
    text file: txt file located root/starter/model/
        Metrics from sliced categortical groups.
    """

    with open(output_path, "w") as f:
        for feature in categorical_features:
            f.write(f"Feature: {feature}\n")
            values = data[feature].unique()

            for val in values:
                slice_df = data[data[feature] == val]

                if slice_df.empty:
                    continue

                X_slice, y_slice, _, _ = process_data(
                    slice_df,
                    categorical_features=categorical_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )

                preds = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(y_slice, preds)

                f.write(
                    f"  {feature} = {val} | "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"Fbeta: {fbeta:.4f}\n"
                )

            f.write("\n")
