import	socket
import threading
from queue import Queue
import time
data_rec=Queue()
host="127.0.0.3"
port=1239

s=socket.socket()
s.connect((host,port))

def Main():
	print ('yo')
	while True:
		#s.send(str.encode(str(message)))
		data=s.recv(1024)
		data=data.decode()
		if not data:
			break
		data_rec.put(data)
		#print("received"+str(data))
		#message=input("->")
	s.close()
#if __name__==('__main__'):
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
	#receive data routine
    if data_rec.qsize()>0:
        print(data_rec.get())
