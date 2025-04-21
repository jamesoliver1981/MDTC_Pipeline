import os
import shutil

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

def clear_tmp_data(tmp_dir="data/tmp"):
    """
    Deletes all contents of the temporary data directory.

    Args:
        tmp_dir (str): Path to the tmp folder (default is 'data/tmp')
    """
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"[utils] Cleared temporary data in: {tmp_dir}")
    else:
        print(f"[utils] No tmp directory found at: {tmp_dir} — nothing to clear.")