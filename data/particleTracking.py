import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from glob import glob 

'''
VIDEO INFO
Format: QuickTime / MOV
Source: ExperimentalVideo.mp4
Duration: 59.900 seconds
Frame rate: 30.017 fps
Length: 1798 frames
Frame Shape: (1440, 1024, 3)
'''

mpl.rc('image', cmap='gray')

video = 'ExperimentalVideo.mp4'
frameidx = 1000

@pims.pipeline
def gray(image):
    return image[:, :, 1]

def track(video, frameidx, minmass=200):
    frames = pims.PyAVVideoReader(video)
    frame = gray(frames[frameidx])

    pixels = 9

    x_start = 70
    x_end = 925
    y_start = 0
    y_end = 1440
    frame_cropped = frame[y_start:y_end, x_start:x_end]
    
    f = tp.locate(frame_cropped, pixels, invert=False, minmass=minmass)
    f['x'] = f['x'] + x_start
    return f, frame



def track_points(video, frameidx):
    f, frame = track(video, frameidx)

    x = f['x'].values
    y = f['y'].values
    
    data = []
    data.append({"x": x, "y": y})

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    f, frame = track(video, frameidx)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    tp.annotate(f, frame, ax=axs[0])
    axs[1].imshow(frame)
    axs[0].set_title(f'particles identified = {len(f)}')
    plt.tight_layout()
    plt.show()