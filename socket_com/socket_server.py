import socket
import time
def Main():
    host="10.42.0.1"
    port=1234	
    s=socket.socket()
    s.bind((host,port))
    
    s.listen(1)
    C,addr = s.accept()
    
	send=False
	if send:
    	while message !='q':
        	C.send(str.encode(str(message)))
        	data=C.recv(1024)
        	print("received"+str(data))
        	message=input("->")
	else:
		while True:
			data=C.recv(1024)
			print('received'+str(data))

    s.close()


if __name__=='__main__':
    Main()
