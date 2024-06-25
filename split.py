# take UVG dataset and split into 24 patches
import os
import cv2 as cv

height = 1080
width = 1920

x_patches = 6
y_patches = 4

assert height % y_patches == 0
assert width % x_patches == 0

patch_h = height // y_patches
patch_w = width // x_patches

path = "../UVG"
outpath = f"./UVG_{patch_h}.{patch_w}"
os.mkdir(outpath)

for video in os.listdir(path):
    video_path = os.path.join(path, video) 
    for i in range(x_patches * y_patches):
        os.mkdir(os.path.join(outpath,f"{video.title()}-{i+1:02d}"))

    for framei, frame_name in enumerate(os.listdir(video_path)):
        frame = cv.imread(os.path.join(video_path, frame_name) )
        h, w, _ = frame.shape
        assert h == height
        assert w == width

        for i in range(y_patches):
            for j in range(x_patches):
                y0, y1 = i * patch_h, (i + 1) * patch_h
                x0, x1 = j * patch_w, (j + 1) * patch_w
                patch = frame[y0:y1, x0:x1]
                
                patch_num = i * x_patches + j + 1
                patch_folder = os.path.join(outpath,f"{video.title()}-{patch_num:02d}")
                patch_filename = os.path.join(patch_folder, f"frame{framei+1:06d}.png")
                print(patch_filename)
                cv.imwrite(patch_filename, patch)