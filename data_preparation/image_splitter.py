import cv2
import os

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    #while vidcap.isOpened():
    while count <11000:
        print(count)
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames('/home/jiyouzhang/data_astro/3rd_team/GH010012_1.mp4', '/home/jiyouzhang/data_astro/3rd_team/image')
