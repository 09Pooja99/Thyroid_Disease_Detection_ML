import os
import sys

class ThyroidException(Exception):
    def __init__(self, error_messege: Exception, error_detail: sys):
        super().__init__(error_messege)
        self.error_messege = ThyroidException.get_detailed_error_messege(
            error_messege=error_messege,
            error_detail=error_detail,  # Consistent naming
        )

    @staticmethod
    def get_detailed_error_messege(error_messege: Exception, error_detail: sys) -> str:
        _, _, exec_tb = error_detail.exc_info()
        line_number = exec_tb.tb_lineno  # Get the line number where the exception occurred
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_messege = (
            f"Error occurred in script: [{file_name}] at line number: [{line_number}] "
            f"error message: [{error_messege}]"
        )
        return error_messege

    def __str__(self):
        return self.error_messege

    def __repr__(self) -> str:
        return f"{ThyroidException.__name__}({self.error_messege})"
