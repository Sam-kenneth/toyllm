import sys
import logging

logger = logging.getLogger('src.exceptions.exception')

class PipelineException(Exception):
    """Custom exception for pipeline errors with detailed context."""
    
    def __init__(self, error_message, error_details=None):
        super().__init__(str(error_message))
        
        # Use provided error_details or get current exception info
        if error_details is None:
            error_details = sys
        
        _, _, exc_tb = error_details.exc_info()
        
        if exc_tb:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename 
        else:
            self.lineno = 'N/A'
            self.file_name = 'N/A'
            
        self.error_message = str(error_message)
        logger.error(str(self))
    
    def __str__(self):
        return f"Error occurred in python script name [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}]"