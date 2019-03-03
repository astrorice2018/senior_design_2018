import numpy as np
import os
import shutil
import re

data_dir="/home/jiyouzhang/data_astro/3rd_team/image/"
target_dir='/home/jiyouzhang/data_astro/3rd_team/image/data/'
patch_num=5
output=[]
for i in range(0,patch_num):
    if len(output)==0:
        output=np.load(data_dir+'patch'+str(i)+'/image_det/coord.npy')
        image_order=np.load(data_dir+'patch'+str(i)+'/image_det/image_order.npy')
        y=np.load(data_dir+'patch'+str(i)+'/image_det/order_in_frame.npy')
    else:
        output=np.concatenate((output,np.load(data_dir+'patch'+str(i)+'/image_det/coord.npy')),axis=0)
        image_order=np.concatenate((image_order,np.load(data_dir+'patch'+str(i)+'/image_det/image_order.npy')),axis=0)
        y=np.concatenate((y,np.load(data_dir+'patch'+str(i)+'/image_det/order_in_frame.npy')),axis=0)        
    f_lst=os.listdir(data_dir+'patch'+str(i)+'/image_det/')
    for f in f_lst:
        if 'png' in f:
            shutil.copy(data_dir+'patch'+str(i)+'/image_det/'+f,target_dir)
np.save(target_dir+'/coord.npy',output)
np.save(target_dir+'/image_order.npy',image_order)
np.save(target_dir+'/order_in_frame.npy',y)


