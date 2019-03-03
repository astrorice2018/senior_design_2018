import  socket

def Main():
    host="10.0.0.4"
    port=1234
	send=False
    s=socket.socket()
    s.connect((host,port))

    #message=input("->")

	if send:
    	while message !='q':
        	s.send(str.encode(str(message)))
        	data=s.recv(1024)
        	print("received"+str(data))
        	message=input("->")
	else:
		while True:
			data=s.recv(1024)
			print ('received'+str(data))
    s.close()
if __name__==('__main__'):
	Main()
