from src.logger_manager import LoggerManager

def main():
    """
    Main function to remove all log files using LoggerManager.
    It creates a logger, logs some info, and clears all existing logs.
    """
    # Initialize the LoggerManager (this creates a new log file)
    logger_manager = LoggerManager(console=True)
    logger = logger_manager.get_logger()

    # Log that we are about to clear logs
    logger.info("Starting log cleanup process...")

    # Clear all existing logs
    logger_manager.clear_all_logs()

    # Inform that cleanup is complete
    logger.info("Log cleanup process completed successfully.")

if __name__ == "__main__":
    main()
