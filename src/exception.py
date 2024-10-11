import sys
import traceback
import logging

def error_message_detail(error, error_detail: sys):
    """
    This function captures and returns detailed error information.
    It extracts the error message, file name, and line number where the error occurred.

    Args:
    error: The raised error (exception).
    error_detail: sys module to extract error details.

    Returns:
    A formatted string with the error message, file name, and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name}, line: {line_number}, error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Custom exception class that logs detailed error messages.
        
        Args:
        error_message: The original error message.
        error_detail: sys module to extract error details.
        """
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)


    def __str__(self):
        return self.error_message
# if __name__=="__main__":

#     try:
        
#         1/0
#     except Exception as e:
#      logging.info('is ther an exception')    
#      error_details = error_message_detail(e, sys)
#      print(error_details)


   


# Example usage:
# try:
#     # Example that raises an error
#     1 / 0
# except Exception as e:
#     error_details = error_message_detail(e, sys)
#     print(error_details)
