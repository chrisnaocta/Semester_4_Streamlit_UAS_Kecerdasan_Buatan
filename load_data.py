import pandas as pd
def load_data():
    df = pd.read_csv("dataset/movie_dataset.csv")
    return df