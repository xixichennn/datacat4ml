# conda environment: pyg (Python 3.9.16)
import logging

def setup_logging(log_file, level, include_host=False):
    """
    This function simplifies logging setup, making it easier to debug and monitor programs, especially in complex or distributed environments.
    """

    # include hostname in log format if specified
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    # set global logging level
    logging.root.setLevel(level)
    # adjust levels of exsisiting loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    # add handlers:
    # stream handler(console logging)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    # file handler (file logging)
    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
