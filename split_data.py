from sklearn.model_selection import train_test_split
from preprocessing import preprocess_for_sentiment
import joblib

def split_data(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    X_train = X_train.apply(preprocess_for_sentiment)
    X_test = X_test.apply(preprocess_for_sentiment)

    return X_train, X_test, y_train, y_test