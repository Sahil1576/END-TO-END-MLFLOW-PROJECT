import sys


class CustomException(Exception):

    def __init__(self, error_message):

        _, _, exc_tb = sys.exc_info()

        self.line_number = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

        self.error_message = str(error_message)

        super().__init__(self.error_message)

    def __str__(self):

        return (
            f"Error occurred in Python script "
            f"[{self.file_name}] "
            f"at line number [{self.line_number}] "
            f"with error message [{self.error_message}]"
        )