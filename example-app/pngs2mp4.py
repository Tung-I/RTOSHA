from PIL import Image
import os
import cv2
import numpy as np

def pngs2mp4(pngs_dir, mp4_file):
    fnames = []
    for filename in os.listdir(pngs_dir):
        if filename.endswith('.png'):
            fnames.append(filename)
    fnames.sort()
    
    writer = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
    for filename in fnames:
        im = Image.open(pngs_dir + filename)
        writer.write(cv2.cvtColor(cv2.UMat(np.array(im)), cv2.COLOR_RGB2BGR))
    writer.release()

pngs2mp4('/home/ubuntu/datasets/goha/tungi/', 'tungi.mp4')