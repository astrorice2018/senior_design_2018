import numpy as np
import os
import shutil
import re
dist_thresh=500      #if person move too fast; it will base on this thresh to think if it is a new target (edge case handle)
inactive_thresh=30   #how many frames of inactivity we decide to not consider some target (cache mechanism)
def sort_output(output,image_order,y):
    #replace the indexing with the real indexing 
    temp_indx=0
    output1=np.array(output)
    output1[0,0]=image_order[0]
    for i in range(1,len(output[:,0])):
        if output[i,0]!=output[i-1,0]:
            temp_indx+=1
        output1[i,0]=image_order[temp_indx]
    indx=output1[:,0].argsort()
    output1=output1[indx]
    y=y[indx]
    return output1,y


def norm2(a,b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2

def weed_out(output,data_dir,y):
    #sort_output first
    y_weed=[]
    #remove nonperson entries
    cls=output[:,-1]
    weed_out=np.empty((1,len(output[0,:])),dtype=float)
    if cls[0]==0:
        weed_out[0,:]=output[0,:]
        y_weed.append(y[0])
    for i in range(1,len(cls)):
        if cls[i]==0:
            weed_out=np.concatenate((weed_out,np.reshape(output[i,:],(1,len(output[i,:])))),0)
            y_weed.append(y[i])
    #maybe save npy file to overwrite


    #remove images
    filelist=os.listdir(data_dir)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    for i in filelist:
        if 'person' not in i:
            os.remove(data_dir+i) 
    return weed_out,y_weed


def find_nn(output,target_dir,data_dir,y):
    frame_id=list(range(0,int(output[:,0][-1])+1))
    center_coords=tuple((output[:,1:3]+output[:,3:5])/2)
    pre_id=0
    pool=[]
    pool_coord=[]
    dist_pool=[]
    pool_activity=[]
    filelist=os.listdir(data_dir)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    image_counter=0
    for i in range(0,len(output[:,0])):
        print ('pool_coord: ',pool_coord)
        frame=str(int(output[i,0]))
        #if frame=='3':
        #    exit()
        #print(pool_coord)
        print(center_coords[i])
        #check if iterating in the same frame
        if output[i,0]==pre_id:
            #initial patching
            if pre_id==0:
                try:
                    shutil.rmtree(target_dir+str(len(pool)))
                except:
                    pass
                os.makedirs(target_dir+str(len(pool)))
                #currently assume any objects other than humans will be 
                #put below the humans, so there is no discontinuity in 
                #in the indexing 
                if "0_"+str(y[len(pool)])+"person.png" in filelist:
                    shutil.copy(data_dir+"0_"+str(y[len(pool)])+"person.png",target_dir+str(len(pool))+"/"+"0_"+str(y[len(pool)])+"person.png")
                print ("0_"+str(y[len(pool)])+"person.png")
                pool_coord.append(center_coords[len(pool)])
                pool.append(len(pool))
                pool_activity.append(0)
            else:
                #increase inactivity log by one
                for ac in range(0,len(pool_activity)):
                    pool_activity[ac]+=1;

                #calculate each distance
                dist_pool=[]
                for j in range(0,len(pool_coord)):
                    dist_pool.append(norm2(pool_coord[j],center_coords[i]))
                #upper thresh for movement
                print(min(dist_pool),'   ===  ',frame+"_"+str(y[i])+"person.png")
                if min(dist_pool)<dist_thresh:
                    #which pool to put in
                    nn_indx=np.argmin(dist_pool)
                    #update pool coordinates
                    pool_coord[nn_indx]=center_coords[i]
                    #update inactivity log
                    pool_activity[nn_indx]=0
                    frame=str(int(output[i,0]))
                    shutil.copy(data_dir+frame+"_"+str(y[i])+"person.png",target_dir+str(nn_indx)+"/"+frame+"_"+str(y[i])+"person.png")
                #dude....lost track of image!!!!!
                #new target pool appears
                else:
                    frame=str(int(output[i,0]))
                    try:
                        shutil.rmtree(target_dir+str(len(pool)))
                    except:
                        pass
                    os.makedirs(target_dir+str(len(pool)))
                    #currently assume any objects other than humans will be 
                    #put below the humans, so there is no discontinuity in 
                    #in the indexing 
                    if str(frame)+"_"+str(y[i])+"person.png" in filelist:
                        shutil.copy(data_dir+str(frame)+"_"+str(y[i])+"person.png",target_dir+str(len(pool))+"/"+str(frame)+"_"+str(y[i])+"person.png")
                    pool_coord.append(center_coords[i])
                    pool.append(len(pool))
                    pool_activity.append(0)
            image_counter+=1
        else:
            pre_id=output[i,0]
            image_counter=0
            #calculate each distance
            dist_pool=[]
            for j in range(0,len(pool_coord)):
                dist_pool.append(norm2(pool_coord[j],center_coords[i]))
            #upper thresh for movement
            print (min(dist_pool),'   ===  ',frame+"_"+str(y[i])+"person.png")
            if min(dist_pool)<dist_thresh:
                #which pool to put in
                nn_indx=np.argmin(dist_pool)
                #update pool coordinates
                pool_coord[nn_indx]=center_coords[i]
                pool_activity[nn_indx]=0
                frame=str(int(output[i,0]))
                shutil.copy(data_dir+frame+"_"+str(y[i])+"person.png",target_dir+str(nn_indx)+"/"+frame+"_"+str(y[i])+"person.png")
            #dude....lost track of image!!!!!

            #new target pool appears
            else:
                frame=str(int(output[i,0]))
                try:
                    shutil.rmtree(target_dir+str(len(pool)))
                except:
                    pass
                os.makedirs(target_dir+str(len(pool)))
                #currently assume any objects other than humans will be 
                #put below the humans, so there is no discontinuity in 
                #in the indexing 
                if str(frame)+"_"+str(y[i])+"person.png" in filelist:
                    shutil.copy(data_dir+str(frame)+"_"+str(y[i])+"person.png",target_dir+str(len(pool))+"/"+str(frame)+"_"+str(y[i])+"person.png")
                pool_coord.append(center_coords[i])
                pool.append(len(pool))
                pool_activity.append(0)
            image_counter+=1
            print (pool_activity)
            for ac in range(0,len(pool_activity)):
                if pool_activity[ac]>=inactive_thresh:
                    pool_coord[ac]=(9999999,9999999);

    #print (pool_coord)
data_dir="/home/jiyouzhang/data_astro/3rd_team/image/data/"
target_dir='/home/jiyouzhang/data_astro/3rd_team/image/data/image_det_target/'
output=np.load(data_dir+'coord.npy')
image_order=np.load(data_dir+'image_order.npy')
y=np.load(data_dir+'order_in_frame.npy')
[output,y]=sort_output(output,image_order,y)
[output,y]=weed_out(output,data_dir,y)
np.save(data_dir+'coord.npy',output)
np.save(data_dir+'/order_in_frame.npy',y)
frame_id=list(range(0,int(output[:,0][-1])+1))

#add the center coords

center_coords=(output[:,1:3]+output[:,3:5])/2    
print(y)
find_nn(output,target_dir,data_dir,y)


#have a target pool keept updating
#every target has a folder with its current coords
#new candidate targets will compare with these coords.

#for i in frame_id:
#    if i==0:
#        
#print (center_coords)
