from datetime import datetime


def to_dt(value):
    return datetime(value.year, value.month, value.day, value.hour, value.minute)
