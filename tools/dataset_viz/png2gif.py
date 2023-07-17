import imageio
import os
from PIL import Image

# args needed to specify
folder_path = r'' # path to directory of pngs
save_path = r'' # path to directory to save gif

if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, os.path.basename(folder_path) + '.gif')

fps = 30  # specify the fps of the gif
quality = 15  # specify the quality level of the gif (0-100)

# get the list of png files in the folder
file_list = os.listdir(folder_path)
png_list = [f for f in file_list if f.endswith('.png')]

# sort the png files in ascending order
png_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# read the png files, compress them and create the gif
images = []
for png_file in png_list:
    image_path = os.path.join(folder_path, png_file)
    with Image.open(image_path) as im:
        im = im.convert('RGB')  # convert to RGB mode to support compression
        im = im.resize((im.size[0] // 2, im.size[1] // 2))  # resize to reduce size
        im.save(image_path, optimize=True, quality=quality)  # compress and save
        image = imageio.imread(image_path)
        images.append(image)

# save the gif
imageio.mimsave(save_path, images, fps=fps)