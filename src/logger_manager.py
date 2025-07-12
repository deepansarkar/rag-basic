import os
import logging
from datetime import datetime

class LoggerManager:
    """
    LoggerManager sets up a logger that writes to a uniquely named log file
    inside a 'logs' directory. It also optionally writes to the console.

    Each time you create a LoggerManager, it creates a new log file named
    with the full datetime stamp, e.g. logs/2025-07-12_17-10-55.log.

    The logs are formatted like:
    [2025-07-12 17:10:55] [INFO] Starting process...
    """

    def __init__(self, base_log_dir='logs', console=False):
        """
        Initializes the logger system.

        Parameters:
        - base_log_dir (str): Directory where logs will be stored (default: 'logs').
        - console (bool): If True, also prints logs to stdout.

        This ensures the log directory exists, creates a uniquely named log file,
        and sets up logging with the desired format.
        """
        try:
            self.base_log_dir = base_log_dir
            self.console = console

            # Ensure that the base log directory exists (creates it if it doesn't)
            os.makedirs(self.base_log_dir, exist_ok=True)

            # Create a unique log filename based on the current datetime
            # Example: logs/2025-07-12_17-10-55.log
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file_path = os.path.join(self.base_log_dir, f"{timestamp}.log")

            # Set up a logger with a unique name to avoid conflicts
            self.logger = logging.getLogger(f"RAGLogger-{timestamp}")
            self.logger.setLevel(logging.DEBUG)  # Capture all levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

            # Important: Avoid duplicate handlers if logger is created multiple times
            if not self.logger.handlers:
                # File handler writes logs to the created file
                file_handler = logging.FileHandler(self.log_file_path, mode='a')
                # Set log format: [2025-07-12 17:10:55] [LEVEL] message
                formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

                # Optional console logging
                if self.console:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)

            # Log that the logger has been initialized
            self.logger.info(f"Logger initialized. Logging to file: {self.log_file_path}")

        except Exception as e:
            # Fallback: if anything fails (disk full, permission error), still log to console
            print(f"[ERROR] Failed to initialize logger: {e}")
            self.logger = logging.getLogger("FallbackLogger")
            self.logger.setLevel(logging.DEBUG)
            fallback_handler = logging.StreamHandler()
            fallback_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                                            datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(fallback_handler)
            self.logger.error("Logger fell back to console only due to initialization error.")

    def get_logger(self):
        """
        Returns the configured logger object.
        Use this to call logger.info(), logger.error(), etc.
        """
        return self.logger

    def get_log_file_path(self):
        """
        Returns the full path to the current log file.
        Example: logs/2025-07-12_17-10-55.log
        """
        return self.log_file_path

    def clear_all_logs(self):
        """
        Removes all log files in the current logs directory.

        Notes:
        - Only removes files ending with '.log' inside the base_log_dir.
        - Skips non-log files (if any) and prints a summary of removed files.

        Example log output:
        [INFO] Removed log file: logs/2025-07-12_17-10-55.log
        [INFO] Removed log file: logs/2025-07-12_17-20-12.log
        [INFO] All log files have been cleared.
        """
        try:
            removed_files = 0

            # Loop over all files in the log directory
            for filename in os.listdir(self.base_log_dir):
                file_path = os.path.join(self.base_log_dir, filename)

                # Only target .log files
                if filename.endswith(".log") and os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_files += 1
                    if self.logger:
                        self.logger.info(f"Removed log file: {file_path}")

            if self.logger:
                if removed_files == 0:
                    self.logger.info("No log files found to remove.")
                else:
                    self.logger.info(f"All log files have been cleared ({removed_files} files).")

        except Exception as e:
            # If something goes wrong, log the error
            if self.logger:
                self.logger.error(f"Error while clearing logs: {e}")
            else:
                print(f"[ERROR] Could not clear logs: {e}")
