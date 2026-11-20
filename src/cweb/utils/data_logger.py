import os
import pickle
import pandas as pd

class DataLogger:
    def __init__(self, folder):
        """
        Creates a data logger for a specific folder
        
        Args: 
            folder (str) : The folder to be used.
        """
        self.folder = folder

        os.makedirs(folder, exist_ok=True)

    def save(self, subpath : str, object : any) -> None:
        path = os.path.join(self.folder, subpath)
        with open(path, 'wb') as file:
            pickle.dump(object, file)

    def load(self, subpath : str) -> any:
        path = os.path.join(self.folder, subpath)
        with open(path, 'rb') as file:
            return pickle.load(file)

    def save_df(self, subpath : str, df : pd.DataFrame) -> None:
        path = os.path.join(self.folder, subpath)
        df.to_csv(path)

    def load_df(self, subpath : str) -> pd.DataFrame:
        path = os.path.join(self.folder, subpath)
        return pd.read_csv(path)

