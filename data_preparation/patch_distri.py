import numpy as np
import os
import shutil
import re


data_dir="/home/jiyouzhang/data_astro/3rd_team/image/"
target_dir='/home/jiyouzhang/data_astro/3rd_team/image/'
f_lst=os.listdir(data_dir)
i_lst=[]

for f in f_lst:
    if 'png' in f:
        i_lst.append(f)
    elif 'patch' in f:
        shutil.rmtree(target_dir+f)
#exit()
counter=0
for i in i_lst:
    num=str(int(int(re.findall(r'\d+',i)[0])/100))
    if not os.path.exists(target_dir+'patch'+num):
        os.makedirs(target_dir+'patch'+num)
    shutil.move(data_dir+i,target_dir+'patch'+num+'/'+i)

