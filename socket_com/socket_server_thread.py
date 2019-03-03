import socket
import time
import threading
from queue import Queue
data_rec=Queue()
host="127.0.0.3"
port=1239
s=socket.socket()
s.bind((host,port))

s.listen(1)
C,addr=s.accept()


def Main():

	while True:
		 data=C.recv(1024)
		 data=data.decode()
		 if not data:
			 break
		 data_rec.put(data)
		 #print ("from client"+str(data))
		 #C.send(data.encode()) 
		# data=str(data).upper()
		#
		# print("sending"+str(data))
		#C.send(i.to_bytes(2, byteorder='big'))
		#C.send(timestamp.to_bytes(8,byteorder='big'))
		#i=(i+1)% 1000
		#timestamp=timestamp+1
		#time.sleep(0.5)

	C.close()


#if __name__=='__main__':
#	 Main()


def threader():
	while True:
		Main()
t=threading.Thread(target=threader)
t.daemon=True
t.start()


#######################################################
#copy the line above to the top of ur code
#the line below is ur code
#


#it is a example here 


while True:
    time.sleep(1)
	#send data routine
    C.send(str.encode(str(8888)))



