from __future__ import print_function
import logging

import grpc

import decryptor_pb2
import decryptor_pb2_grpc


def sendCiphertext(cipher_t, bytes):
    with grpc.insecure_channel('172.25.244.5:50050') as channel:#open channel

        stub = decryptor_pb2_grpc.DecryptorStub(channel)#create stub

        response = stub.Decrypt(decryptor_pb2.GetDecryptionRequest(ciphertext = cipher_t,key_filename = bytes))
        #create vaild request message
        #print("plain text is : " + response.plaintext)
        return response.plaintext




if __name__ == '__main__':
    logging.basicConfig()
    f = open("data0.enc", 'rb')
    data = f.read()
    f.close()
    sendCiphertext(data)    
