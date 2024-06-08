## given one folder, concat the images in them and save them in the same folder
import os
import cv2 
import numpy as np 
import imageio 
def concat_images(folder,name="generated_images.jpg",policy='square'):
    files = os.listdir(folder)
    files.sort()
    images=[]
    for f in files:
        if f[-3:] in ['jpg','png'] and name not in f:
            images.append(cv2.imread(os.path.join(folder,f)))
    num = len(images)
    height, width, layers = images[0].shape
    if policy == "square":
        # select 9,16,25 images 
        images = images[:9]
        # images = np.concatenate(images,axis=1).reshape(height*3, width*num, layers)
        big_image = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)

        # 将每张小图片放置到大图像数组中的相应位置
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                big_image[i*height:(i+1)*height, j*width:(j+1)*width, :] = images[idx]
                
        images =big_image
        # print(images.shape,type(images))
    else:
        images = np.concatenate(images,axis=1).reshape(height, width*num, layers)
    cv2.imwrite(os.path.join(folder,name),images)

## image to gifs

def images2gif(image_files:list,save_path:str):
    gif_frames = []
    for file_name in image_files:
        # print(file_name)
        gif_frames.append(imageio.imread(file_name))
    imageio.mimsave(save_path, gif_frames, duration=0.5) 

if __name__=="__main__":
    
    folder = "./sample/mnist/"
    folder = "./sample/randn/"
    name = "generated_images.jpg"
    subfolders = sorted(os.listdir(folder))
    # for sf in subfolders:
    #     os.system(f"mv {os.path.join(folder,sf)} {os.path.join(folder,f'{int(sf):05d}')}")
    # subfolders = sorted(os.listdir(folder))
    
    for sf in subfolders:
        concat_images(os.path.join(folder,sf),name,'square')
    
    image_files = sorted([os.path.join(folder,sf,name) for sf in subfolders])
    images2gif(image_files,os.path.join(folder,"generated_images.gif"))