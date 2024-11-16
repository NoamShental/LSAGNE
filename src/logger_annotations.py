from enum import Enum, auto


class LogAnnotations:
    MICHAEL_ONLY = 'michael_only'
    MICHAEL_NOT_INTERESTED = 'michael_not_interested'


MICHAEL_ONLY_ANNOTATION = {LogAnnotations.MICHAEL_ONLY: True}

MICHAEL_NOT_INTERESTED_ANNOTATION = {LogAnnotations.MICHAEL_NOT_INTERESTED: True}
