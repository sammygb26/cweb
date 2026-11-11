import os

LOG_FILE = 'log.txt'

class Logger:
    def __init__(self, folder : str):
        """
        Creates a logger for a specific folder.

        Args:
            folder (str) : The folder to be used.
        """
        self.folder = folder
        self.path = os.path.join(folder, LOG_FILE)

        os.makedirs(folder, exist_ok=True)

    def log(self, line : str) -> None:
        """
        Logs something to the log file.

        Args:
            string (str) : S tring to be logged.
        """
        with open(self.path, 'a+') as file:
            file.write(line + '\n')

