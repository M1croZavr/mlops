# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import grpc
import ml_service_pb2 as ml__service__pb2

GRPC_GENERATED_VERSION = "1.68.1"
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + " but the generated code in ml_service_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class MLServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.LoadData = channel.unary_unary(
            "/MLService/LoadData",
            request_serializer=ml__service__pb2.LoadDataRequest.SerializeToString,
            response_deserializer=ml__service__pb2.LoadDataResponse.FromString,
            _registered_method=True,
        )
        self.TrainModel = channel.unary_unary(
            "/MLService/TrainModel",
            request_serializer=ml__service__pb2.TrainModelRequest.SerializeToString,
            response_deserializer=ml__service__pb2.TrainModelResponse.FromString,
            _registered_method=True,
        )
        self.Predict = channel.unary_unary(
            "/MLService/Predict",
            request_serializer=ml__service__pb2.PredictRequest.SerializeToString,
            response_deserializer=ml__service__pb2.PredictResponse.FromString,
            _registered_method=True,
        )


class MLServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def LoadData(self, request, context):
        """Load data for training and validation"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def TrainModel(self, request, context):
        """Train the model"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Predict(self, request, context):
        """Predict by the model"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_MLServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "LoadData": grpc.unary_unary_rpc_method_handler(
            servicer.LoadData,
            request_deserializer=ml__service__pb2.LoadDataRequest.FromString,
            response_serializer=ml__service__pb2.LoadDataResponse.SerializeToString,
        ),
        "TrainModel": grpc.unary_unary_rpc_method_handler(
            servicer.TrainModel,
            request_deserializer=ml__service__pb2.TrainModelRequest.FromString,
            response_serializer=ml__service__pb2.TrainModelResponse.SerializeToString,
        ),
        "Predict": grpc.unary_unary_rpc_method_handler(
            servicer.Predict,
            request_deserializer=ml__service__pb2.PredictRequest.FromString,
            response_serializer=ml__service__pb2.PredictResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "MLService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers("MLService", rpc_method_handlers)


# This class is part of an EXPERIMENTAL API.
class MLService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def LoadData(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/MLService/LoadData",
            ml__service__pb2.LoadDataRequest.SerializeToString,
            ml__service__pb2.LoadDataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def TrainModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/MLService/TrainModel",
            ml__service__pb2.TrainModelRequest.SerializeToString,
            ml__service__pb2.TrainModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def Predict(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/MLService/Predict",
            ml__service__pb2.PredictRequest.SerializeToString,
            ml__service__pb2.PredictResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
