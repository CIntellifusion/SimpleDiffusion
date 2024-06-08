## given one folder, concat the images in them and save them in the same folder
import os
import cv2 
import numpy as np 
import imageio 
import importlib
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


### config functions 

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if  "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))