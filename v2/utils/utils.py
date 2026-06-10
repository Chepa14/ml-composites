import pandas as pd


def read_csv(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, engine="python")
    # df = pd.read_csv(TXT_PATH, sep=r"\s+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df
