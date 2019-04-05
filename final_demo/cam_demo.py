from __future__ import division
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
import os 
import shutil
import re
data_dir='/home/luke/'
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
	if str(label)=='person':
		print('person!!!!')
		cv2.imwrite(data_dir+str(frame_id)+'_'+str(y)+'person_test.png',img[c1[1]:c2[1],c1[0]:c2[0],:])
	#color = random.choice(colors)
	#cv2.rectangle(img, c1, c2,color, 1)
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	#cv2.rectangle(img, c1, c2,color, -1)
	#cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
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
				cv2.imshow("frame", orig_im)
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					break
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

			flst=os.listdir(data_dir)
			waitreid=True 
			if ('status') in flst:
				#check if reid is done 
				while waitreid:
					with open(data_dir+'/'+'status') as file:
						txtlines=file.readlines()
						for line in txtlines:
							if 'reid_busy' in line:
								waitreid= (re.findall(r'\d+',line)[0]=='1')
				with open(data_dir+'/'+'status','r') as file:
					txtlines=file.readlines()
					for line in range(0,len(txtlines)):
						if 'bb_busy' in txtlines[line]:
							txtlines[line]='bb_busy=1\n' 
				open(data_dir+'/'+'status', 'w').close()
				with open(data_dir+'/'+'status','w') as file:
					file.writelines(txtlines)
			else:
				print ('status file missing')
				exit()

			for filenm in flst:
				if filenm.endswith('.png'):
					os.remove(data_dir+'/'+filenm)
			x_vt=(output[:,1]+output[:,3])/2
			y_vt=output[:,4]
			x_vt=np.reshape(x_vt,(len(x_vt),1))
			y_vt=np.reshape(y_vt,(len(y_vt),1))
			np.save(data_dir+'/coord.npy',np.concatenate((x_vt,y_vt),1))
			list(map(lambda x,y: write(x, orig_im,frame_id,y), output,y1))
			frame_id+=1
			if frame_id==2:
				exit()
			#no need to keep this, because it is only showing the frame
			#but the parameters may be useful for the use elsewhere
 
			cv2.imshow("frame", orig_im)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				break
			frames += 1
			print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
			with open(data_dir+'/'+'status','r') as file:
				txtlines=file.readlines()
				for line in range(0,len(txtlines)):
					if 'bb_busy' in txtlines[line]:
						txtlines[line]='bb_busy=0\n'
			open(data_dir+'/'+'status', 'w').close()
			with open(data_dir+'/'+'status','w') as file:
				file.writelines(txtlines)
			exit()
			
		else:
			break
	

	
	

