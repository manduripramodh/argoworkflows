import logging

from http import HTTPStatus

from request_utils import get_request_id, get_target
from error_codes import OwnErrorCode, SubComponents, ERROR_MESSAGES

TRACKING_PREFIX_CODE = "02"


class TrackingException(Exception):
    """
    Generic Explainable AI Api Exception class
    """

    def __init__(
            self,
            status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
            error_code: OwnErrorCode = OwnErrorCode.GENERIC_API_ERR,
            subcomponent: SubComponents = SubComponents.OWN_SUBCODE_PREFIX,
            message: str = None,
            message_fields: list = None,
            details: list = None
    ):
        super().__init__()
        self.status_code = status_code
        self.subcomp_prefix = subcomponent
        error_data = TrackingException.generate_error_data(
            error_code, self.subcomp_prefix, message_fields)
        self.error_code = error_data['code']
        self.message = message if message else error_data['message']
        self.target = get_target()
        self.request_id = get_request_id()
        self.details = details

    def __str__(self):
        return "{} : {} : {}".format(self.status_code, self.error_code, self.message)

    @staticmethod
    def generate_error_data(code, subcomp_code, message_fields):
        if not isinstance(code, OwnErrorCode) or not isinstance(subcomp_code, SubComponents):
            logging.getLogger(__name__).info("Exception: Error code or subcomponent code doesn't belong to Enum class")
            code = OwnErrorCode.GENERIC_API_ERR
        try:
            msg = ERROR_MESSAGES[code]
        except KeyError:
            code = OwnErrorCode.GENERIC_API_ERR
            msg = ERROR_MESSAGES[code]
        return {'code': TRACKING_PREFIX_CODE + subcomp_code.value + code.value,
                'message': msg.format(message_fields) if message_fields else msg
                }
