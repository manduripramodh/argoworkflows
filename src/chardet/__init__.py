import logging

# This will throw warning, as in the requests library,
# they are checking the compatibility of urllib3 needs chardet version.
# chardet >= 3.0.2, < 3.1.0 for urllib3 >= 1.21.1, <= 1.25
__version__ = '1.0.0'


# returning the encoding utf-8 by default
def detect(byte_string):
    logger = logging.getLogger("ml_tracking.chardet")
    logger.debug("Using explainable AI chardet to guess the encoding")
    result = {'encoding': 'utf-8'}
    return result
