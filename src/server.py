import connexion
from flask import jsonify
from connexion.exceptions import ProblemException
from error_codes import OwnErrorCode
from http import HTTPStatus
from exceptions import TrackingException
from request_utils import get_request_id, get_target

class Server:
    def __init__(self):
        self.app = connexion.FlaskApp(__name__, specification_dir='swagger')
        self.app.add_api(specification='main-v2.yaml')
        for error_code in [400, 403, 404, 405, 409, 500]:
            self.app.add_error_handler(error_code, self.__handle_exception)
        self.app.add_error_handler(Exception, self.__handle_exception)
        self.app.add_error_handler(ProblemException, self.handle_connexion_exception)

    def get_app(self):
        return self.app

    @staticmethod
    def healthz():
        return 'OK', 200

    @staticmethod
    def handle_connexion_exception(error):
        problem_exception = TrackingException(status_code=HTTPStatus.BAD_REQUEST,  # pylint: disable=no-member
                                              error_code=OwnErrorCode.BAD_REQEST_ERR, message=error.detail)
        return Server.__handle_exception(problem_exception)

    @staticmethod
    def __handle_exception(error):

        if hasattr(error, "request_id"):
            request_id = error.request_id
        else:
            request_id = get_request_id()

        if hasattr(error, "error_code"):
            error_code = error.error_code
        else:
            error_code = 500
        error_map = {
            400: 'Invalid Request',
            403: 'Forbidden',
            404: 'URL does not exist',
            405: 'Invalid method for URL',
            409: 'Conflict',
            500: 'Internal Server Error',
            503: 'Service Unavailable'
        }
        if hasattr(error, "message"):
            message = error.message
        else:
            message = error_map[error_code]
        error_object = {
            'code': str(error_code),
            'message': message,
            'requestId': request_id,
            'target': get_target()
        }
        if hasattr(error, "details"):
            error_object['details'] = error.details
        response = jsonify(error=error_object)
        if hasattr(error, "status_code"):
            response.status_code = error.status_code
        else:
            response.status_code = error_code
        return response

if __name__ == '__main__':
    Server().get_app().run(port=5000)
