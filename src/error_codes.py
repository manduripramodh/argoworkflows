"""
Module containing various error codes and messages which tracking api uses
"""
from enum import Enum, unique
"""
class defining own submodule specific error codes
"""
@unique
class OwnErrorCode(Enum):
    GENERIC_API_ERR = "0000"
    GENERIC_SERVICE_ERR = "0001"
    BAD_REQEST_ERR = "0002"
    NOT_FOUND_ERR = "0003"
    INVALID_FILTER_ERR = "0004"
    PAYLOAD_LIMIT_ERR = "0005"
    FORBIDDEN_ERR = "0006"
    DB_CONN_ERR = "0007"
    DB_CRED_ERR = "0008"
    ILLEGAL_ARGS_ERR = "0009"
    INVALID_FIND_FILTER_QUERY = "0010"
    RESP_PAYLOAD_LIMIT_ERR = "0011"
    MISSING_ARG_FIND_ERR1 = "0012"
    MISSING_ARG_FIND_ERR2 = "0013"
    MISSING_ARG_FIND_ERR3 = "0014"
    MISSING_ARG_FIND_ERR4 = "0015"
    PERSISTED_RUN_NOT_FOUND_ERR = "0016"
    CREATE_MISSING_ARG_ERR1 = "0017"
    CREATE_MISSING_ARG_ERR2 = "0018"
    CREATE_MISSING_ARG_ERR3 = "0019"
    CREATE_MISSING_ARG_ERR4 = "0020"
    DELETION_MISSING_ARG_ERR1 = "0021"
    DELETION_MISSING_ARG_ERR2 = "0022"
    DELETION_MISSING_ARG_ERR3 = "0023"
    DELETION_ILLEGAL_ARGS_ERR = "0024"
    DELETION_RUN_NOT_FOUND = "0025"
    INVALID_JSON_PAYLOAD = "0026"
    EMPTY_PATCH_PAYLOAD_ERR = "0027"
    ILLEGAL_DB_ATTRIBUTE = "0028"
    INVALID_FILTER_QUERY_FIND_ERR = "0029"
    INVALID_EXPAND_QUERY = "0030"
    QUERIED_RUN_NOT_FOUND = "0031"
    RUN_NOT_FOUND__PATCH_ERR = "0032"
    INVALID_DELETION_REQ = "0033"
    DATETIME_HANDLING_ERR = "0034"
    NONE_TYPE_METRICS_ERR = "0035"

@unique
class AuditErrorCode(Enum):
    AUDIT_POST_ERR = "0036"


# class defining various subcomponents of  error codes



@unique
class SubComponents(Enum):
    OWN_SUBCODE_PREFIX = "00"
    DB_SUBCODE_PREFIX = "01"
    AUDIT_SUBCODE_PREFIX = "02"


# dictionary object defining various error message string mapping to the error codes

ERROR_MESSAGES = {
    OwnErrorCode.GENERIC_API_ERR: "TrackingException",
    OwnErrorCode.GENERIC_SERVICE_ERR: "ServiceUnavailableException",
    OwnErrorCode.BAD_REQEST_ERR: "BadRequestException",
    OwnErrorCode.NOT_FOUND_ERR: "NotFoundException",
    OwnErrorCode.INVALID_FILTER_ERR: "InvalidFilterException: {}",
    OwnErrorCode.PAYLOAD_LIMIT_ERR: "PayloadLimitException",
    OwnErrorCode.FORBIDDEN_ERR: "ForbiddenException",
    OwnErrorCode.DB_CONN_ERR: "DBConnectionException",
    OwnErrorCode.DB_CRED_ERR: "DBCredentialsProviderError: {}",
    OwnErrorCode.ILLEGAL_ARGS_ERR: "IllegalArgumentException",
    OwnErrorCode.INVALID_FIND_FILTER_QUERY: "Invalid filter query {}",
    OwnErrorCode.RESP_PAYLOAD_LIMIT_ERR: "Response payload exceeded. Drill down further using ExecutionId/RunId to \
reduce response payload",
    OwnErrorCode.MISSING_ARG_FIND_ERR1: "Either notebookId/scenarioId or runId or pipelineId/executionId required",
    OwnErrorCode.MISSING_ARG_FIND_ERR2: "Either notebookId or pipelineId/executionId should be provided",
    OwnErrorCode.MISSING_ARG_FIND_ERR3: "ScenarioId/RunId must be provided to get metric from the notebookId",
    OwnErrorCode.MISSING_ARG_FIND_ERR4: "scenarioId must be provided along with scenarioVersion",
    OwnErrorCode.PERSISTED_RUN_NOT_FOUND_ERR: "Run was not found",
    OwnErrorCode.CREATE_MISSING_ARG_ERR1: "Either notebookId or pipelineId/executionId required",
    OwnErrorCode.CREATE_MISSING_ARG_ERR2: "Either strictly notebookId or pipelineId/executionId should be provided",
    OwnErrorCode.CREATE_MISSING_ARG_ERR3: "scenarioVersion must be provided along with isVersionDirty flag",
    OwnErrorCode.CREATE_MISSING_ARG_ERR4: "Either metrics or params or both should be provided",
    OwnErrorCode.DELETION_MISSING_ARG_ERR1: "request not allowed without parameters [scenarioId scenarioVersion \
isVersionDirty notebookId runId executionId pipelineId]",
    OwnErrorCode.DELETION_MISSING_ARG_ERR2: "scenarioId/runId should be provided with notebookId",
    OwnErrorCode.DELETION_MISSING_ARG_ERR3: "scenarioId must be provided with scenarioVersion",
    OwnErrorCode.DELETION_ILLEGAL_ARGS_ERR: "No other arguments can be provided with a list of scenario ids",
    OwnErrorCode.DELETION_RUN_NOT_FOUND: "Run not found",
    OwnErrorCode.INVALID_JSON_PAYLOAD: "Invalid json payload. Please provide resource(runCollection tags)",
    OwnErrorCode.EMPTY_PATCH_PAYLOAD_ERR: "Empty tags & runCollection not allowed",
    OwnErrorCode.ILLEGAL_DB_ATTRIBUTE: "Illegal attribute: {}",
    OwnErrorCode.INVALID_FILTER_QUERY_FIND_ERR: "Invalid filter query {}",
    OwnErrorCode.INVALID_EXPAND_QUERY: "Invalid expand query",
    OwnErrorCode.QUERIED_RUN_NOT_FOUND: "Run was not found with given filter",
    OwnErrorCode.RUN_NOT_FOUND__PATCH_ERR: "Run was not found",
    OwnErrorCode.INVALID_DELETION_REQ: "deletion not allowed without parameters[scenarioId scenarioVersion \
isVersionDirty notebookId runId executionId pipelineId]",
    OwnErrorCode.DATETIME_HANDLING_ERR: "error handling datetime: {}",
    OwnErrorCode.NONE_TYPE_METRICS_ERR: "Metric of none type value is not allowed",
    AuditErrorCode.AUDIT_POST_ERR: "Error posting to audit log end point"
}
