import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.clear_session()
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Concatenate,Flatten,Reshape,Cropping2D, Conv3D,Conv3DTranspose,MaxPool3D,Cropping3D, Reshape,Lambda
from tensorflow.python.keras.models import Model
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import os
import util





#------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------

COCO_BODY_PARTS = ['nose', 'neck',
                   'right_shoulder', 'right_elbow', 'right_wrist',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_hip', 'left_knee', 'left_ankle',
                   'right_eye', 'left_eye', 'right_ear', 'left_ear', 'background'
                   ]

def create_model_SRSF_nodepthguidance():
    UB = True
    (h1,h2,h3,h4) = (3,3,3,3)
    (w1,w2,w3,w4) = (3,3,3,3)
    (N1, N2, N3, N4) = (64,32,16,8)
    acti_fct = 'relu'
    input_histogram = Input((4,4,100,1))


    deconv1 = Conv3DTranspose(N1, (h1,h1,w1), padding='same',strides = (1,1,1), activation=acti_fct,use_bias=UB)(input_histogram) # 8x8x100
    print(deconv1)
    conv11 = Conv3D(N1, (h1,h1,w1), padding='same', activation=acti_fct,use_bias=UB)(deconv1) # 8x8x100
    conv12 = Conv3D(N1, (h1,h1,w1), padding='same', activation=acti_fct,use_bias=UB)(conv11) # 8x8x100

    max_pool1 = MaxPool3D(pool_size=(1, 1, 2), strides=None, padding='same')(conv12) # 8x8x50
    deconv2 = Conv3DTranspose(N2, (h2,h2,w2), padding='same', strides = (2,2,1),activation=acti_fct,use_bias=UB)(max_pool1) # 16x16x50
    conv21 = Conv3D(N2, (h2,h2,w2), padding='same', activation=acti_fct,use_bias=UB)(deconv2) # 16x16x50
    conv22 = Conv3D(N2, (h2,h2,w2), padding='same', activation=acti_fct,use_bias=UB)(conv21) # 16x16x50
    max_pool2 = MaxPool3D(pool_size=(1, 1, 2), strides=None, padding='same')(conv22)# 16x16x25
    deconv3 = Conv3DTranspose(N3, (h3,h3,w3), padding='same', strides = (2,2,1), activation=acti_fct,use_bias=UB)(max_pool2) # 32x32x25
    conv31 = Conv3D(N3, (h3,h3,w3), padding='same', activation=acti_fct,use_bias=UB)(deconv3) # 32x32x25
    conv32 = Conv3D(N3, (h3,h3,w3), padding='same', activation=acti_fct,use_bias=UB)(conv31) # 32x32x25
    max_pool3 = MaxPool3D(pool_size=(1, 1, 2), strides=None, padding='same')(conv32) # 32x32x13
    deconv4 = Conv3DTranspose(N4, (h4,h4,w4), padding='same', strides = (2,2,1), activation=acti_fct,use_bias=UB)(max_pool3) # 64x64x13x64
    conv41 = Conv3D(N4, (h4,h4,w4), padding='same', activation=acti_fct,use_bias=UB)(deconv4) # 64x64x13
    conv42 = Conv3D(N4, (h4,h4,w4), padding='same', activation=acti_fct,use_bias=UB)(conv41) # 64x64x13
    max_pool4 = MaxPool3D(pool_size=(1, 1, 2), strides=None, padding='same')(conv42) # 64x64x7
    reshape_4 = Reshape(( max_pool4.shape[1],max_pool4.shape[2], max_pool4.shape[3]*max_pool4.shape[4]))(max_pool4)
    conv_last = Conv2D(1, (3,3), padding='same', activation=acti_fct,use_bias=UB)(reshape_4)
    model = Model(input_histogram,conv_last)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = "mean_squared_error", metrics = ["accuracy"])
    return model
def predict_heat_pafs_3stage():
    input_depth = Input((32,32,1))
    # Predict Heatmaps
    conv1_depth = Conv2D(100, (3,3), padding="same", activation='relu')(input_depth) # 32,32
    maxpool1_depth = MaxPool2D(pool_size=(2,2))(conv1_depth) 
    conv2_depth = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_depth) # 16 16  
    maxpool2_depth = MaxPool2D(pool_size=(2,2))(conv2_depth)
    conv3_depth =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_depth) # 8 8 
    conv4_depth = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_depth) # 16 16 
    conv5_depth = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_depth) # 32 32
    heatmaps_std1 = Conv2D(18, (3,3), padding="same", activation='relu')(conv5_depth) # 8 8 
    print(heatmaps_std1.shape)

    # Predict PAFS
    conv1_pafs = Conv2D(100, (3,3), padding="same", activation='relu')(input_depth) # 32,32
    maxpool1_pafs = MaxPool2D(pool_size=(2,2))(conv1_pafs) 
    conv2_pafs = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_pafs) # 16 16  
    maxpool2_pafs = MaxPool2D(pool_size=(2,2))(conv2_pafs)
    conv3_pafs =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_pafs) # 8 8 
    conv4_pafs = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_pafs) # 16 16 
    conv5_pafs = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_pafs) # 32 32
    pafs_std1 = Conv2D(38, (3,3), padding="same", activation='relu')(conv5_pafs) # 8 8 
    print(pafs_std1.shape)

    # concatenate Both
    both = Concatenate(axis=-1)([heatmaps_std1, pafs_std1]) # Nbacth, 32, 32, 25+14
     
    # Predict Heatmaps stage2
    conv1_depth_st2 = Conv2D(100, (3,3), padding="same", activation='relu')(both) # 32,32
    maxpool1_depth_st2 = MaxPool2D(pool_size=(2,2))(conv1_depth_st2) 
    conv2_depth_st2 = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_depth_st2) # 16 16  
    maxpool2_depth_st2 = MaxPool2D(pool_size=(2,2))(conv2_depth_st2)
    conv3_depth_st2 =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_depth_st2) # 8 8 
    conv4_depth_st2 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_depth_st2) # 16 16 
    conv5_depth_st2 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_depth_st2) # 32 32
    heatmaps_st2 = Conv2D(18, (3,3), padding="same", activation='relu')(conv5_depth_st2) # 8 8 
    print(heatmaps_st2.shape)

    # Predict PAFS
    conv1_pafs_st2 = Conv2D(100, (3,3), padding="same", activation='relu')(both) # 32,32
    maxpool1_pafs_st2 = MaxPool2D(pool_size=(2,2))(conv1_pafs_st2) 
    conv2_pafs_st2 = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_pafs_st2) # 16 16  
    maxpool2_pafs_st2 = MaxPool2D(pool_size=(2,2))(conv2_pafs_st2)
    conv3_pafs_st2 =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_pafs_st2) # 8 8 
    conv4_pafs_st2 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_pafs_st2) # 16 16 
    conv5_pafs_st2 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_pafs_st2) # 32 32
    pafs_st2 = Conv2D(38, (3,3), padding="same", activation='relu')(conv5_pafs_st2) # 8 8 
    print(pafs_st2.shape)
    both2 = Concatenate(axis=-1)([heatmaps_st2, pafs_st2])
    
    # Predict Heatmaps stage3
    conv1_depth_st3 = Conv2D(100, (3,3), padding="same", activation='relu')(both2) # 32,32
    maxpool1_depth_st3 = MaxPool2D(pool_size=(2,2))(conv1_depth_st3) 
    conv2_depth_st3 = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_depth_st3) # 16 16  
    maxpool2_depth_st3 = MaxPool2D(pool_size=(2,2))(conv2_depth_st3)
    conv3_depth_st3 =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_depth_st3) # 8 8 
    conv4_depth_st3 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_depth_st3) # 16 16 
    conv5_depth_st3 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_depth_st3) # 32 32
    heatmaps_st3 = Conv2D(18, (3,3), padding="same", activation='relu')(conv5_depth_st3) # 8 8 
    print(heatmaps_st3.shape)

    # Predict PAFS
    conv1_pafs_st3 = Conv2D(100, (3,3), padding="same", activation='relu')(both2) # 32,32
    maxpool1_pafs_st3 = MaxPool2D(pool_size=(2,2))(conv1_pafs_st3) 
    conv2_pafs_st3 = Conv2D(100, (3,3), padding="same", activation='relu')(maxpool1_pafs_st3) # 16 16  
    maxpool2_pafs_st3 = MaxPool2D(pool_size=(2,2))(conv2_pafs_st3)
    conv3_pafs_st3 =  Conv2D(100, (3,3), padding="same", activation='relu')(maxpool2_pafs_st3) # 8 8 
    conv4_pafs_st3 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv3_pafs_st3) # 16 16 
    conv5_pafs_st3 = Conv2DTranspose(100, (3,3), padding="same", strides = (2,2), activation='relu')(conv4_pafs_st3) # 32 32
    pafs_st3 = Conv2D(38, (3,3), padding="same", activation='relu')(conv5_pafs_st3) # 8 8 
    print(pafs_st2.shape)
    both3 = Concatenate(axis=-1)([heatmaps_st3, pafs_st3])

    output = Concatenate(axis=-1)([both,both2,both3]) # Nbacth, 32, 32, (25+14)*2

    model = Model(input_depth,output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = "mean_squared_error", metrics = ["accuracy"])
    return model
def extract_parts(heatmap_avg,paf_avg,  input_image_shape, params, model_params):
    #multiplier = [x * model_params['boxsize'] / input_image.shape[0] for x in params['scale_search']]

    # Body parts location heatmap, one per part (19)
    #heatmap_avg = np.zeros((input_image.shape[0], input_image.shape[1], 19))
    # Part affinities, one per limb (38)
    #paf_avg = np.zeros((input_image.shape[0], input_image.shape[1], 38))
    


    all_peaks = []
    peak_counter = 0

    for part in range(18):
        hmap_ori = heatmap_avg[:, :, part]
        hmap = gaussian_filter(hmap_ori, sigma=3)

        # Find the pixel that has maximum value compared to those around it
        hmap_left = np.zeros(hmap.shape)
        hmap_left[1:, :] = hmap[:-1, :]
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:-1, :] = hmap[1:, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:, 1:] = hmap[:, :-1]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[:, :-1] = hmap[:, 1:]

        # reduce needed because there are > 2 arguments
        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_left, hmap >= hmap_right, hmap >= hmap_up, hmap >= hmap_down, hmap > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        
        peaks_with_score = [x + (hmap_ori[x[1], x[0]],) for x in peaks]  # add a third element to tuple with score
        idx = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (idx[i],) for i in range(len(idx))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(util.hmapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in util.hmapIdx[k]]]
        cand_a = all_peaks[util.limbSeq[k][0] - 1]
        cand_b = all_peaks[util.limbSeq[k][1] - 1]
        n_a = len(cand_a)
        n_b = len(cand_b)

        # index_a, index_b = util.limbSeq[k]
        if n_a != 0 and n_b != 0:
            connection_candidate = []
            for i in range(n_a):
                for j in range(n_b):
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                        np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * input_image_shape / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connection) >= min(n_a, n_b):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = np.empty((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(util.hmapIdx)):
        if k not in special_k:
            part_as = connection_all[k][:, 0]
            part_bs = connection_all[k][:, 1]
            index_a, index_b = np.array(util.limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][index_b] != part_bs[i]:
                        subset[j][index_b] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][index_b] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    delete_idx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)
    points = []
    for peak in all_peaks:
        try:
            points.append((peak[0][:2]))
        except IndexError:
            points.append((None, None))
    body_parts = dict(zip(COCO_BODY_PARTS, points))
    return body_parts, all_peaks, subset, candidate
def draw_3d(postpro, depth_image, subset, candidate):
    
    # Inputs

    # depth_image 
    # all_peaks, subset, candidate : from extract_parts(heatmap_avg_1,paf_avg_1,..) with heatmap_avg_1 and paf_avg_1 of corresponding to the depth images
    # correction_depth : postprocessing to correct completely out depth
    # resize_fac=1

    # Outputs 

    ##############################
    # 1. CALCULATE DEPTH
    ##############################

    depth_map = np.zeros((len(subset),candidate.shape[0]))
    for limb in range(17):
        for index_person in range(len(subset)):
            index = subset[index_person, np.array(util.limbSeq[limb]) - 1]
            if -1 in index:
                continue
            y = candidate[index.astype(int), 0]
            x = candidate[index.astype(int), 1]
            z_0 = depth_image[int(x[0]), int(y[0])]
            z_1 = depth_image[int(x[1]), int(y[1])]
            depth_map[int(index_person), int(index[0])] = z_0
            depth_map[int(index_person), int(index[1])] = z_1

    


    #############################################
    # 2. COMPENSATE FOR DEPTH FAR FROM MEDIAN
    #############################################
    for index_person in range(len(subset)):
        depth_1 = depth_map[int(index_person), :] 
        depth_cleaned = depth_1[depth_1>120]
        median_depth = np.median(depth_cleaned)
        for i in range(candidate.shape[0]):
            if np.abs(depth_map[int(index_person), i] - median_depth) > 5:
                depth_map[index_person, i]  = median_depth 


    #############################################
    # 3. COMPENSATE FOR MAGNIFICATION OF LENS
    #############################################
    candidate_3D = np.empty((candidate.shape[0], 3))
    for limb in range(17):
        for index_person in range(len(subset)):
            index = subset[index_person, np.array(util.limbSeq[limb]) - 1]
            if -1 in index:
                continue
            for indexindex in index:
                #print('STEP 3')
                y = candidate[int(indexindex), 0]
                x = candidate[int(indexindex), 1]
                #print(x)
                z = depth_map[int(index_person), int(indexindex)]
                #new_x = (x - 100)*z/120 + 100
                #new_y = (y - 110)*z/120 + 110
                new_x = (x - 100)*z/167 + 100
                new_y = (y - 110)*z/167 + 110
                candidate_3D[int(indexindex), 0] = new_x
                candidate_3D[int(indexindex), 1] = new_y
                candidate_3D[int(indexindex), 2] = z
                
    #############################################
    # 4. RETURN LINES
    #############################################
    lines = []
    corresponding_color = []
    for limb in range(17):
        #print('limb'+str(limb))
        index_bis = 0
        for s in subset:
            #print('limbSeq'+str(np.array(util.limbSeq[limb])))
            index = s[np.array(util.limbSeq[limb]) - 1]
            #print('index'+str(index))
            if -1 in index:
                continue

            x = candidate_3D[index.astype(int), 0]
            #print('x'+str(x))
            y = candidate_3D[index.astype(int), 1]
            #print('y'+str(y))
            z = candidate_3D[index.astype(int), 2]

            
            lines.append([(y[0],z[0], 223-x[0]), (y[1], z[1], 223-x[1])])
            
            corresponding_color.append(z[0]/300)
            
            index_bis =  index_bis + 1

    return candidate_3D, lines,corresponding_color
def function_scale_color(col):
    val = 0.25
    # y = -x + 1 + val (val index ou on veut commencer bleu->0.25)
    if isinstance(col, list):
        tab = [1 for i in range(len(col)) if  i < val]
        tab = [-i+1+val for i in col if i>val]
    else :
        if col<val:
            tab = 1
        else:
            tab = -col+1 +val
    return tab
