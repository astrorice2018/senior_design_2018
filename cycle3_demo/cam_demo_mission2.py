#search 'reid' to see what lines needs to be modfied for future demo

from __future__ import division
import socket 
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl

ip='127.0.0.1' #ip address of the server 
port_num=1234  #port to connect
data_dir='/home/luke/' #bb output location; no use here will be used for reid 
#u need to uncomment a lot of code for enable majority vote at panda side
max_window=100         #if u want handle majority vote at panda side it matters
threshold=50        #if 50 out of 100 frames with detected person send actions
frame_count=0
target_hit=0
sc=True             #True: panda acts as server; False: panda acts as client;

def get_test_input(input_dim, CUDA):
	img = cv2.imread("imgs/messi.jpg")
	img = cv2.resize(img, (input_dim, input_dim)) 
	img_ =	img[:,:,::-1].transpose((2,0,1))
	img_ = img_[np.newaxis,:,:,:]/255.0
	img_ = torch.from_numpy(img_).float()
	img_ = Variable(img_)
	
	if CUDA:
		img_ = img_.cuda()
	
	return img_

def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network. 
	
	Returns a Variable 
	"""

	orig_im = img
	dim = orig_im.shape[1], orig_im.shape[0]
	img = cv2.resize(orig_im, (inp_dim, inp_dim))
	img_ = img[:,:,::-1].transpose((2,0,1)).copy()
	img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
	return img_, orig_im, dim

def write(x, img,frame_id,y):
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	#if str(label)=='person':
		#print('person!!!!')
		#we dont need to save images for now, but later for reid we have to
		#cv2.imwrite(data_dir+str(frame_id)+'_'+str(y)+'person_test.png',img[c1[1]:c2[1],c1[0]:c2[0],:])
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	return img

def arg_parse():
	"""
	Parse arguements to the detect module
	
	"""
	
	
	parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
	parser.add_argument("--reso", dest = 'reso', help = 
						"Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
						default = "160", type = str)
	return parser.parse_args()



if __name__ == '__main__':
	host=ip
	port=port_num
	s=socket.socket()
	if sc:
		s.bind((host,port))
		s.listen(1)
	print ('connecting.......')
	if sc:
		C,addr=s.accept()
		so=C
	else:
		s.connect((host,port))
		so=s
	print ("================================")
	print('Socket connection established !!!')
	print ("================================")
	

	print ('')
	#print ('Waiting to start bb: s (start) q (quit)')
	#message=input("->")
	#if message =='s':
#		pass
#	elif message=='q':
#		exit()
#	else:
#		print ('wrong input quit')
#		exit()

	cfgfile = "cfg/yolov3.cfg"
	weightsfile = "yolov3.weights"
	num_classes = 80

	args = arg_parse()
	confidence = float(args.confidence)
	nms_thesh = float(args.nms_thresh)
	start = 0
	CUDA = torch.cuda.is_available()
	

	
	
	num_classes = 80
	bbox_attrs = 5 + num_classes
	
	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	
	model.net_info["height"] = args.reso
	inp_dim = int(model.net_info["height"])
	
	assert inp_dim % 32 == 0 
	assert inp_dim > 32

	if CUDA:
		model.cuda()
			
	model.eval()
	
	videofile = 'video.avi'
	
	cap = cv2.VideoCapture(0)
	
	assert cap.isOpened(), 'Cannot capture source'
	
	frames = 0
	start = time.time()
	frame_id=0	  
	while cap.isOpened():
		#loop control
		
		ret, frame = cap.read() #get the image
		if ret:
			
			img, orig_im, dim = prep_image(frame, inp_dim)
			
#			 im_dim = torch.FloatTensor(dim).repeat(1,2)						
			
			
			if CUDA:
				im_dim = im_dim.cuda()
				img = img.cuda()
			
			
			#output control
			output = model(Variable(img), CUDA)
			output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
			
			#no need to keep this, because it is only showing the frame
			#but the parameters may be useful for the use elsewhere
			if type(output) == int:
				frames += 1
				print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
				#cv2.imshow("frame", orig_im)
				#key = cv2.waitKey(1)
				#if key & 0xFF == ord('q'):
				#	break
				continue
			

		
			output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
			
#			 im_dim = im_dim.repeat(output.size(0), 1)
			output[:,[1,3]] *= frame.shape[1]
			output[:,[2,4]] *= frame.shape[0]

			
			classes = load_classes('data/coco.names')
			colors = pkl.load(open("pallete", "rb"))
			img_id=output[:,0]
			y1=[0]
			for i in range(1,len(img_id)):
				if int(img_id[i])==int(img_id[i-1]):
					y1.append(y1[i-1]+1)
				else:
					y1.append(0)
	
			if frame_count==0:
				target_hit=0
        # uncomment the following lines to enable majority vote from panda
		#	if target_hit==threshold:
		#		while True:
		#			so.send (str.encode("target appear! land!"))
		#			time.sleep(0.5)
			
			frame_count=(frame_count+1)%max_window
		#the following code is to see if the frame is a mushy frame
		#if the bb is size zero at (0,0,0,0) and only has one target it is mush
		#frame		
			c1 = tuple(output[0][1:3].int())
			c2 = tuple(output[0][3:5].int())
			if (len(orig_im[:,0,0])*0.9<=c2[1]-c1[1]) or (len(orig_im[0,:,0])*0.9<=c2[0]-c1[0]) or (c2[1]==0 and c2[0]==0 and c1[0]==0 and c1[1]==0) and len(output)==1:
				bb_p=False
			else:
				bb_p=True
			print('img_size: ', len(orig_im[:,0,0]))
			print('bb size: ',c2,c1)
			print ('target name:', output[:,-1])
			print ('target num', len(output)==1)
			
			#update of target_hit should be in write() function if reid
			labels0=output[:,-1]
			#comment out so.send lines to enable majority vote
			if (torch.tensor([0.]) in labels0) and bb_p:
				print ('person in frame')
				target_hit+=1
				so.send(str.encode('1'))
			else:
				print ('no person')	
				so.send(str.encode('2'))
			#uncomment the following line for reid 
			#list(map(lambda x,y: write(x, orig_im,frame_id,y), output,y1))
			frame_id=(frame_id+1)%10
			#no need to keep this, because it is only showing the frame
			#but the parameters may be useful for the use elsewhere
 
			#cv2.imshow("frame", orig_im)
			#key = cv2.waitKey(1)
			#if key & 0xFF == ord('q'):
			#	break
			frames += 1
			print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

			
		else:
			break
