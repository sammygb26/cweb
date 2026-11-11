import os
import pickle
import jax.numpy as jnp

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

