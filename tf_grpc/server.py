from concurrent import futures
import logging

import grpc

import decryptor_pb2_grpc
import decryptor_pb2

sample_string = 'plain_text'

class Decryptor(decryptor_pb2_grpc.DecryptorServicer):

    def Decrypt(self, request, context):
        print("Decrypt request signal:",request.ciphertext)
        return decryptor_pb2.GetDecryptionResponse(plaintext=sample_string)



def make_plain_text_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    decryptor_pb2_grpc.add_DecryptorServicer_to_server(Decryptor(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    make_plain_text_server()
