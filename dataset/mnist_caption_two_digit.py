import h5py
import numpy as np
# import pylab
import datetime
import scipy
# from scipy.misc import toimage
import numpy as np
import random
from random import randint
np.random.seed(np.random.randint(1<<30))




num_frames = 10
seq_length = 10
image_size = 64
batch_size = 1
num_digits = 2
step_length = 0.1
digit_size = 28
frame_size = image_size ** 2


def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}
    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse


dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'and': 12, 'are':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'around':18, 'jumping':19, 'up':20, 'down':21, '.':22, 'is':23, 'left':24, 'right': 25}


motion_strings = ['left and right', 'up and down']
motion_idxs = np.array([[0,0],[0,1],[1,0],[1,1]])

def create_dataset():
    numbers = np.random.permutation(100)
    dataset = np.zeros((4, 10 * 10), dtype = np.int)
    dataset[0,:] = numbers
    dataset[1,:] = 100 + numbers
    dataset[2,:] = 200 + numbers
    dataset[3,:] = 300 + numbers
    train = []
    val = []
    count = 0 
    for i in range(100):
        dummy = count % 2
        val.append(dataset[dummy, i])
        train.append(dataset[1-dummy, i])
        count = count + 1
    for i in range(100):
        dummy = count % 2
        val.append(dataset[dummy + 2, i])
        train.append(dataset[(1 - dummy) + 2, i])
        count = count + 1
    return np.array(train), np.array(val) 

def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words))))

    for i in xrange(len(words)):
        m[0,i] = dictionary[words[i]]
    return m

def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in xrange(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]
    return text


def GetRandomTrajectory(batch_size, motion):
    length = seq_length
    canvas_size = image_size - digit_size
    
    y = np.random.rand(batch_size) # the starting point of the two numbers
    x = np.random.rand(batch_size)

    start_y = np.zeros((length, batch_size)) # 20x128
    start_x = np.zeros((length, batch_size)) # 20x128

    if motion == 0:
        theta = np.ones(batch_size) * 0.5 * np.pi
    else:
        theta = np.zeros(batch_size) 

    v_y = 2 * np.sin(theta)
    v_x = 2 * np.cos(theta)

    for i in xrange(length):
        y += v_y * step_length
        x += v_x * step_length
        # Do not bounce off the edges
        for j in xrange(batch_size):
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
        #print x, y
        start_y[i, :] = y
        start_x[i, :] = x

    #scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    print start_y.shape
    return start_y, start_x

def Overlap(a, b):
    return np.maximum(a, b)

def create_gif(digit_imgs, motion):
    # get an array of random numbers for indices
    start_y1, start_x1 = GetRandomTrajectory(batch_size, motion[0])
    start_y2, start_x2 = GetRandomTrajectory(batch_size, motion[1])
    gifs = np.zeros((seq_length, batch_size, image_size, image_size), dtype = np.float32)
    start_y, start_x = np.concatenate([start_y1, start_y2], axis = 1), np.concatenate([start_x1, start_x2], axis = 1)
    print start_x.shape, start_y.shape
    for j in xrange(batch_size):
        for n in xrange(num_digits):
            digit_image = digit_imgs[n,:,:]
            for i in xrange(num_frames):
                top = start_y[i, j * num_digits + n]
                left = start_x[i, j * num_digits + n]
                bottom = top + digit_size
                right = left + digit_size
                gifs[i, j, top:bottom, left:right] = Overlap(gifs[i, j, top:bottom, left:right], digit_image)
    return gifs

def create_gifs_for_data(dataset, data, labels, num):
    final_gif_data = np.zeros((num, num_frames, 1, image_size, image_size), dtype = np.float32)
    sentence_length = 14
    captions = np.zeros([num, sentence_length], dtype = np.int)
    counts_of_sentences = np.zeros(len(dataset), dtype = np.int)
    outer_index = 0
    inner_digits = dataset % 100
    digits1 = dataset % 10
    digits2 = dataset / 10
    digits = np.concatenate([digits1, digits2])
    motion_values = dataset / 100
    while outer_index < num:

        idxs = np.random.randint(data.shape[0], size = num_digits)
        if labels[idxs[0]] in digits and labels[idxs[1]] in digits:
            n = 10*labels[idxs[0]] + labels[idxs[1]]
            motion_list = np.where(inner_digits == n)[0]
            random.shuffle(motion_list)
            motion_idx = motion_idxs[motion_values[motion_list[0]]]
#            print motion_strings[motion_idx[0]]
#            motion1 == motion_strings[motion_idx[0]]
#            motion2 == motion_strings[motion_idx[1]]

            digit = data[idxs]
            dummy = create_gif(digit, motion_idx)

            final_gif_data[outer_index, :, :, :, :] = dummy
            sentence = 'digit %d is %s and digit %d is %s .' % (labels[idxs[0]], motion_strings[motion_idx[0]], labels[idxs[1]], motion_strings[motion_idx[1]])
            counts_of_sentences[motion_values[motion_list[0]]] += 1
            sentence_matrix = sent2matrix(sentence, dictionary)
            captions[outer_index,:] = sentence_matrix
        else:
            outer_index -= 1
        outer_index += 1

        if outer_index == 10:
            break
    return final_gif_data, captions, counts_of_sentences

f = h5py.File('mnist.h5')
train_data = f['train'].value.reshape(-1, 28, 28)
train_labels = f['train_labels'].value
val_data = f['validation'].value.reshape(-1,28,28)
val_labels = f['validation_labels'].value
f.close()

data = np.concatenate((train_data,val_data), axis = 0)
labels = np.concatenate((train_labels,val_labels), axis = 0)

train, val = create_dataset()

train, val = create_dataset()
# print train, val
data_train, captions_train, count_train = create_gifs_for_data(train, data, labels, 10000)
data_val, captions_val, count_val = create_gifs_for_data(val, data, labels, 2000)


with h5py.File('mnist_two_gif.h5','w') as hf:
    # final_gif_data,captions,counts_of_sentences = create_gifs_for_data(data,labels)
    hf.create_dataset('mnist_gif_train', data=data_train)
    hf.create_dataset('mnist_captions_train', data=captions_train)
    hf.create_dataset('mnist_count_train', data=count_train)
    hf.create_dataset('mnist_dataset_train', data=train)

    hf.create_dataset('mnist_gif_val', data=data_val)
    hf.create_dataset('mnist_captions_val', data=captions_val)
    hf.create_dataset('mnist_count_val', data=count_val)
    hf.create_dataset('mnist_dataset_val', data=val)


