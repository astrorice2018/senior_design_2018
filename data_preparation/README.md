notes:
1.every time run auto label make sure the data is originally generated by detect code (aka u have never run auto label on the data before)!!!!!
2.change pre_id number in auto_label code for different patches

1.run .sh file to make video 30fps   (need to change file location in the code)
2.run image_spliter.py to convert mp4 to images (need to change file location in the code to where u want to store the imagesi; the number of images has to be the mutiple of 100)
3.run patch_distri.py (data location: where the images are stored; target location: where u want to store patches; each of them contain 100 images) 
4.run gen_det_data.sh from !!!pytorch folder!!! (put it there) (location:where the patch is; as arguement for command line); meanwhile put detect1.py in pytorch folder
5.run patch_merger.py (data location: patchs location; target location: where u want to put the giant merged patch at)
6.run auto label  (need to change file location in the code to the giant merged patch)


notes: right now auto_label_edge.py works for our need but may be hard to read. will document it and update it if we require more from this 
