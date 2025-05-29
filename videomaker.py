# importing libraries
import os
import cv2
from PIL import Image
import time
import playvideo as pv
global selection_image
import glob
import shutil
import os

def generate_image_seqences(chara,text):
    selection_image=[]
    image_folder = "./Dataset/"+chara+"/Original"
    images = []
    for filename in os.listdir(image_folder):
        impath = os.path.join(image_folder,filename)
        if impath is not None:
            images.append(impath)
    selection_image.append(images[0])
    print("selection_image=",selection_image)
    return selection_image
    


   


def generate_video(path):
                image_folder = path # make sure to use your folder
                print(image_folder)
                
                # setting the frame width, height width
                # the width, height of first image
                
                # Appending the images to the video one by one
                images = [img for img in os.listdir(image_folder)if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")]
                frame = cv2.imread(os.path.join(image_folder, images[0]))
                height, width, layers = frame.shape
                video = cv2.VideoWriter("./generated/"+path+".avi", 0, 1, (width, height))
                for image in images:
                    video.write(cv2.imread(os.path.join(image_folder, image)))
                # Deallocating memories taken for window creation
                cv2.destroyAllWindows()
                video.release() # releasing the video generated
                # Calling the generate_video function

def process(text):
    selection_image1=[]
    for i in text:
        if i.isalpha():
            chara=i.upper()
            path = "./Dataset/"+chara+"/Original"
            selection_image=generate_image_seqences(chara,text)
            selection_image1.append(selection_image)
    os.mkdir(text)
    for img in selection_image1:
        inv_num = str(img).replace(r'\\', '/')
        print("source imagepath==",img)
        #print("source imagepath==",inv_num)
        file = str(inv_num)
        print("Filepath==",file)
        file=file.replace("'","")
        file=file.replace("[","")
        file=file.replace("]","")
        
        print("Filepath==",file)
        filename=os.path.basename(file).split('/')[-1]
        print("Filename==",filename)
        dst_dir = text
        shutil.copy(file, dst_dir)

    generate_video(text)
    print("Video Generated")
    pv.process(text+".avi")
    time.sleep(0.5)
    
process("apple")
        
