from logging import Logger, getLogger


def get_logger(name: str) -> Logger:
    """
    Creates a logger object for a given name.

    Args:
        name (str): Name of the logger.

    Returns:
        Logger: Logger object with given name.
    """

    logger = getLogger(name)
    return logger