# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import decryptor_pb2 as decryptor__pb2


class DecryptorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Decrypt = channel.unary_unary(
                '/examples.decryptor_server.Decryptor/Decrypt',
                request_serializer=decryptor__pb2.GetDecryptionRequest.SerializeToString,
                response_deserializer=decryptor__pb2.GetDecryptionResponse.FromString,
                )


class DecryptorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Decrypt(self, request, context):
        """Translates the given word.

        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DecryptorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Decrypt': grpc.unary_unary_rpc_method_handler(
                    servicer.Decrypt,
                    request_deserializer=decryptor__pb2.GetDecryptionRequest.FromString,
                    response_serializer=decryptor__pb2.GetDecryptionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'examples.decryptor_server.Decryptor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Decryptor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Decrypt(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/examples.decryptor_server.Decryptor/Decrypt',
            decryptor__pb2.GetDecryptionRequest.SerializeToString,
            decryptor__pb2.GetDecryptionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
