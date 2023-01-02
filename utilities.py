from time import time
from datetime import timedelta


def get_elapsed_time(time_start):
    return timedelta(seconds=round(time() - time_start))
