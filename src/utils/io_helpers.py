import os

def save_dataframe(df, name, folder="data/outputs", index=False):
    """
    Saves a DataFrame to a CSV in the specified output folder.

    Args:
        df (pd.DataFrame): The dataframe to save.
        name (str): Filename without extension.
        folder (str): Directory to save to.
        index (bool): Whether to include the index in the CSV.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"Saved: {path}")