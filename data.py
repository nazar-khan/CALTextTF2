import pickle as pkl
import numpy as np
import cv2

def dataIterator(feature_file, label_file, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # Use with statement to ensure files are closed after use
    with open(feature_file, 'rb') as fp:
        features = pkl.load(fp)
    
    with open(label_file, 'rb') as fp2:
        labels = pkl.load(fp2)
    
    imageSize = {uid: fea.shape[0]*fea.shape[1] for uid, fea in features.items()}
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])  # Sorted by image size
    
    feature_batch, label_batch, feature_total, label_total, uidList = [], [], [], [], []
    batch_image_size, biggest_image_size, i = 0, 0, 0

    for uid, size in imageSize:
        #if size > biggest_image_size:
        #    biggest_image_size = size
        fea = features[uid]
        lab = labels[int(uid)] 

        #batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print(f'sentence {uid} length bigger than {maxlen}, ignoring')
        elif size > maxImagesize:
            print(f'image {uid} size bigger than {maxImagesize}, ignoring')
        else:
            uidList.append(uid)
            #if batch_image_size > batch_Imagesize or i == batch_size:
            if i==batch_size:
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                #biggest_image_size = size
                feature_batch = [fea]
                label_batch = [lab]
                #batch_image_size = biggest_image_size * (i + 1)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # Append the last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print(f'total {len(feature_total)} batch data loaded')

    return list(zip(feature_total, label_total)), uidList

'''
def dataIterator(feature_file, label_file, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # Use with statement to ensure files are closed after use
    with open(feature_file, 'rb') as fp:
        features = pkl.load(fp)
    
    with open(label_file, 'rb') as fp2:
        labels = pkl.load(fp2)
    
    imageSize = {uid: fea.shape[0]*fea.shape[1] for uid, fea in features.items()}
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])  # Sorted by image size
    
    feature_batch, label_batch, feature_total, label_total, uidList = [], [], [], [], []
    batch_image_size, biggest_image_size, i = 0, 0, 0

    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = labels[int(uid)] 

        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print(f'sentence {uid} length bigger than {maxlen}, ignoring')
        elif size > maxImagesize:
            print(f'image {uid} size bigger than {maxImagesize}, ignoring')
        else:
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = [fea]
                label_batch = [lab]
                batch_image_size = biggest_image_size * (i + 1)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # Append the last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print(f'total {len(feature_total)} batch data loaded')

    return list(zip(feature_total, label_total)), uidList
'''

def prepare_data(images_x, seqs_y): #, n_words_src=30000, n_words=30000):
    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y)# + 1
    x = np.zeros((n_samples, max_height_x, max_width_x), dtype='float32')
    y = np.zeros((maxlen_y, n_samples), dtype='int64')  # <eol> should be 0 in dict
    x_mask = np.zeros((n_samples, max_height_x, max_width_x), dtype='float32')
    y_mask = np.zeros((maxlen_y, n_samples), dtype='float32')
    
    for idx, (s_x, s_y) in enumerate(zip(images_x, seqs_y)):
        # Convert to grayscale if necessary
        if len(s_x.shape) > 2:
            s_x = cv2.cvtColor(s_x.astype('float32'), cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        x[idx, :heights_x[idx], :widths_x[idx]] = (s_x - s_x.min()) / (s_x.max() - s_x.min())
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.
    
    return x, x_mask, y, y_mask

def preprocess_img(img):
    # Convert to grayscale if necessary
    if len(img.shape) > 2:
        img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
    
    height, width = img.shape
    if width < 300:
        result = np.ones([height, width * 2]) * 255
        result[:, width:width * 2] = img
        img = result
    
    if height > 300:
        img = img[:300, :]
    
    img = cv2.resize(img, dsize=(800, 100), interpolation=cv2.INTER_AREA)
    
    xx_pad = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
    xx_pad[:] = (img - img.min()) / (img.max() - img.min())
    xx_pad = xx_pad[None, :, :]  # Add batch dimension
    
    return img, xx_pad

