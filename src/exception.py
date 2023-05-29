import sys

def error_message(error, error_detail):
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame