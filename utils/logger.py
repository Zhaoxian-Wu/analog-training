import logging

from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """
    Helper class for logging.
    Write the log into a file and a 
    Arguments:
        log_dir (str): Path to log file.
    """
    def __init__(self, log_dir, log_in_file=True):
        self.log_in_file = log_in_file
        self.log_dir = log_dir
        if log_in_file:
            self.writer = SummaryWriter(log_dir=log_dir)
            event_write_file_name = self.writer.file_writer.event_writer._file_name

            print (f'Logging to file: {event_write_file_name}')
        else:
            event_write_file_name = None
        self.logger = self._get_logger(log_dir, event_write_file_name, log_in_file)
        
    def _get_logger(self, log_dir, event_write_file_name, log_in_file=True):
        # if need to log in file, the ``event_write_file_name`` should not be None
        assert event_write_file_name is not None or not log_in_file
        logger = logging.getLogger(log_dir)
        
        # set up the logger
        logger.setLevel(logging.DEBUG)
        
        # Define the log format
        format_str = '[%(asctime)s] %(message)s'
        # format_str = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        
        # Create a console output handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        # Add handlers to the logger
        logger.addHandler(console_handler)
        
        # Create a file output handler
        if log_in_file:
            file_handler = logging.FileHandler(event_write_file_name + ".log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            # Add handlers to the logger
            logger.addHandler(file_handler)
        
        return logger

    def write(self, step, message, tb_dict):
        # tensorboard record
        if self.log_in_file:
            for key, value in tb_dict.items():
                self.writer.add_scalar(key, value, step)
        # terminal / file record
        if message is not None:
            self.logger.info(message)
