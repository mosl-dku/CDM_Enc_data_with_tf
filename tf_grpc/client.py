from __future__ import print_function
import logging

import grpc

import decryptor_pb2
import decryptor_pb2_grpc


def sendCiphertext(bytes):
    with grpc.insecure_channel('localhost:50051') as channel:#open channel

        stub = decryptor_pb2_grpc.DecryptorStub(channel)#create stub

        response = stub.Decrypt(decryptor_pb2.GetDecryptionRequest(ciphertext = bytes))
        #create vaild request message

        print("plain text is : " + response.plaintext)
        #result




if __name__ == '__main__':
    logging.basicConfig()
    f = open("enc_data", 'rb')
    data = f.read()
    f.close()
    sendCiphertext(data)    
