import uuid
import logging

from connexion import request


REQUEST_ID_HEADER = "X-Request-Id"
BASE_PATH_HEADER = 'X-Original-Uri'


def get_request_id():
    """
     Fetches request id from request header

     Returns:
         str -- Request id if 'X-Request-Id' header present, else returns a randomly generated uuid4
     """
    try:
        request_id = request.headers[REQUEST_ID_HEADER]
    except (KeyError, RuntimeError, TypeError):
        logging.getLogger(__name__).warning(
            '{} not found in headers. Generating uuid instead.'.format(REQUEST_ID_HEADER))
        request_id = str(uuid.uuid4())
    return request_id


def get_target():
    """
    Fetches tartget URL from request header

    Returns:
        str -- Target url if 'X-Original-Uri' header present, else returns base path
    """
    try:
        base_path = request.headers[BASE_PATH_HEADER]
    except (KeyError, RuntimeError, TypeError):
        base_path = '/app/ml-tracking'
    return f'{base_path}{request.path}'
