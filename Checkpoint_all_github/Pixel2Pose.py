

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import tensorflow as tf
import scipy.io as sio
tf.keras.backend.clear_session()
import numpy as np
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import util
import matplotlib.cm as cm
import scipy.io as sio
from config_reader import config_reader
import os
import random
from scipy import ndimage
import cv2
from skimage.transform import resize
from sklearn.utils import shuffle
import scipy.io as sio
import numpy as np
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from moviepy.editor import VideoClip
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D
from functions_FINALRESULTS import *
import argparse

if __name__ == '__main__':
#------------------------------------------------------------------------------------------------------------------------
# LOAD DATA 
#------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)

    args = parser.parse_args()
    scenario = args.scenario
    scenario = 1 # Pick 1, 2, 3 for the number of people present in the frame

    path = '/home/ar432/ST_Project/Checkpoint_all/'+str(scenario)+'_PEOPLE'
    histogram_validation = sio.loadmat(os.path.join(path, 'data.mat'))['histogram']
    rgb_image = sio.loadmat(os.path.join(path, 'data.mat'))['reference_RGB']

    #------------------------------------------------------------------------------------------------------------------------
    #LOAD MODELS
    #------------------------------------------------------------------------------------------------------------------------
    checkpoint_pixels2depth = os.path.join(path, 'Pixels2Depth', 'Checkpoint', 'cp.ckpt')
    checkpoint_depth2pose = os.path.join(path, 'Depth2Pose', 'Checkpoint', 'cp.ckpt')


    model = create_model_SRSF_nodepthguidance()
    # print(model.summary())
    model.load_weights(str(checkpoint_pixels2depth))
    model_heatpafs = predict_heat_pafs_3stage()
    model_heatpafs.load_weights(str(checkpoint_depth2pose))

    # (First test to warm up the GPUs)
    depth_validation = model.predict(histogram_validation)  
    model_heatpafs.predict(depth_validation)

    #------------------------------------------------------------------------------------------------------------------------
    # Pixels2Depth
    #------------------------------------------------------------------------------------------------------------------------

    start_time_model = time.time()
    start_time = time.time()
    depth_validation = model.predict(histogram_validation)
    testing_time = time.time()-start_time
    print('SRSF processing in'+str(testing_time))

    #------------------------------------------------------------------------------------------------------------------------
    # Depth2Pose
    #------------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    pred_heatpafs = model_heatpafs.predict(depth_validation)
    testing_time = time.time()-start_time
    print('POSE processed in'+str(testing_time))

    #------------------------------------------------------------------------------------------------------------------------
    # Postpro module
    #------------------------------------------------------------------------------------------------------------------------
    size_image = (rgb_image.shape[0], rgb_image.shape[1])#(223,220)

    index = 0
    start_time = time.time()
    heatmap_avg_1 =  np.squeeze(pred_heatpafs[index, :,:, 56+18+38:56+18+38+18])
    heatmap_avg_1 = cv2.resize(heatmap_avg_1, (size_image[1], size_image[0]))
    paf_avg_1 = np.squeeze(pred_heatpafs[index, :,:, 56+18+38+18:])
    paf_avg_1 = cv2.resize(paf_avg_1, (size_image[1], size_image[0]))
    paf_avg_1 = paf_avg_1*2-1
    depth_image = depth_validation[index]*250
    depth_image =  cv2.resize(depth_image, (size_image[1], size_image[0]))
    input_image_shape_0 = 200
    params, model_params = config_reader()
    postpro = 1 
    # ESTIMATIONS 
    body_parts, all_peaks, subset_est, candidate = extract_parts(heatmap_avg_1,paf_avg_1, input_image_shape_0, params, model_params)
    postpro=1
    candidate_3D_estimated, lines_est,corresponding_color = draw_3d(postpro, depth_image, subset_est, candidate)

    #print(candidate_3D_estimated)

    testing_time = time.time()-start_time
    print('Lines in'+str(testing_time))
    testing_time_withoutloading = time.time()-start_time_model
    print('Total testing time'+str(testing_time_withoutloading))

    #------------------------------------------------------------------------------------------------------------------------
    # Plot of Results
    #------------------------------------------------------------------------------------------------------------------------
    viridis = cm.get_cmap('viridis')
    fig = plt.figure(figsize=(300,100))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(rgb_image)
    ax.set_title('Reference RGB image', fontsize=250)

    ax = fig.add_subplot(1,3,2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.squeeze(depth_validation))
    im.set_clim(0, 1)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical',ticks=[0.33, 0.66, 0.99])
    cbar.ax.set_yticklabels(['1m', '2m', '3m'],{'fontsize':250}) 
    ax.set_title('Output of Pixels2Depth',fontsize=250)
    ax.axis('off')


    ax = fig.add_subplot(1,3,3, projection='3d')
    for limb in range(17):
        array_limbs = np.array(util.limbSeq[limb]) - 1
        for index_person in range(len(subset_est)):
            index = subset_est[index_person, np.array(util.limbSeq[limb]) - 1]
            if -1 in index:
                continue
            for index_index in index:

                x = candidate_3D_estimated[int(index_index), 0]
                y = candidate_3D_estimated[int(index_index), 1]
                z = candidate_3D_estimated[int(index_index), 2]
                
                val = function_scale_color(z/300)

                color_depth = viridis(val)
                ax.scatter(y, z, 223-x, s = 3000, color=color_depth)

    c = viridis(function_scale_color(corresponding_color))


    lc = Line3DCollection(lines_est, linewidths=20,colors=c)
    ax.add_collection3d(lc)

    x = np.arange(0,200)
    ax.set_xlim3d(0, max(x))
    ax.set_xticks([50, 100, 150, 200])
    ax.set_xticklabels(['0.5', '1', '1.5', '2'])
    ax.set_xlabel('y (m)', fontsize=200)


    z = np.arange(0,223)
    ax.set_zlim3d(0, max(z))
    ax.set_zticks([50, 100, 150, 200])
    ax.set_zticklabels(['0.5', '1', '1.5', '2'])
    ax.set_zlabel('x (m)', fontsize=200)

    y = np.arange(0,300)
    ax.set_ylim3d(min(y), max(y))
    ax.set_yticks([ 100, 200,  300])
    ax.set_yticklabels(['1'  ,'2'  ,'3'])
    ax.set_ylabel('z (m)', fontsize=200)

    ax.xaxis.labelpad = 200
    ax.zaxis.labelpad = 200
    ax.yaxis.labelpad = 150

    ax.xaxis.set_tick_params(labelsize=200)
    ax.zaxis.set_tick_params(labelsize=200)
    ax.yaxis.set_tick_params(labelsize=200)
    #ax.axis('off')
    ax.view_init(10,-60)
    ax.set_title('Output of Pixels2Pose',fontsize=250)

    plt.show()

    plt.savefig(os.path.join(path, 'figure.png'))

