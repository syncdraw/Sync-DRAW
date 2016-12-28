import h5py
import numpy as np
from random import randint # import pylab
import datetime
import scipy
# from scipy.misc import toimage
import numpy as np
from random import randint
import random
np.random.seed(np.random.randint(1<<30))

# f = h5py.File('mnist.h5')
# data = f['train'].value.reshape(-1, 28, 28)
# labels = f['train_labels'].value
# f.close()

num_frames = 10
seq_length = 10
image_size = 64
batch_size = 1
num_digits = 1
step_length = 0.1
digit_size = 28
frame_size = image_size ** 2

def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}
    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse


# dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digits': 11, 'and': 12, 'are':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'around':18, 'jumping':19, 'up':20, 'down':21, '.':22}

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digits': 11, 'and': 12, 'is':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'around':18, 'jumping':19, 'up':20, 'down':21, '.':22, 'left':23, 'right':24}

motion_strings = ['left and right', 'up and down']

def create_dataset():
    numbers = np.random.permutation(10)
    dataset = np.zeros((2,10), dtype = np.int)
    dataset[0,:] = numbers
    dataset[1,:] = 10 + numbers
    train = []
    val = []
    # test = []
    count = 0
    for i in range(10):
        dummy = count % 2
        val.append(dataset[dummy, i])
        train.append(dataset[1-dummy, i])
        count = count + 1
#    for i in range(4,8):
#        dummy = count % 2
#        test.append(dataset[dummy, i])
#        train.append(dataset[1-dummy, i])
#        count = count + 1
#    train.extend(dataset[:,10:].flatten())
    return np.array(train), np.array(val)#,np.array(test)


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

def GetRandomTrajectory(batch_size,motion):
    length = seq_length
    canvas_size = image_size - digit_size
    
    y = np.random.rand(batch_size) # the starting point of the two numbers
    x = np.random.rand(batch_size)

    start_y = np.zeros((length, batch_size)) # 20x128
    start_x = np.zeros((length, batch_size)) # 20x128
    # get the valocity
    #theta = * 2 * np.pi
    if motion == 0:
        theta = np.ones(batch_size) * 0.5 * np.pi
    else:
        theta = np.ones(batch_size) * 0 * np.pi

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

    return start_y, start_x


def Overlap(a, b):
    return np.maximum(a, b)

# function to render the final gif
def create_gif(digit_imgs, motion):
    # get an array of random numbers for indices
    start_y, start_x = GetRandomTrajectory(batch_size * num_digits, motion)
    gifs = np.zeros((seq_length, batch_size, image_size, image_size), dtype = np.float32)
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

            

# def create_captions():
def create_gifs_for_data(dataset, data, labels, num):
    # num = 10000#number of samples that are required
    final_gif_data = np.zeros((num, num_frames, 1, image_size, image_size), dtype = np.float32)
    sentence_length = 9
    captions = np.zeros([num, sentence_length], dtype = np.int)
    counts_of_sentences = np.zeros(len(dataset),dtype=np.int32)
    outer_index = 0
    digits = dataset % 10
    motions_values = dataset / 10
    while outer_index < num:
    #we will be creating a dataset of two numbers, can be increased to three or more
        idxs = np.random.randint(data.shape[0], size = num_digits)
        print outer_index
        if labels[idxs[0]] in digits:
            
            motion_list = np.where(digits==labels[idxs[0]])[0]
            # print motion_list,digits,motions_values,labels[idxs[0]]
            random.shuffle(motion_list)
            motion_idx = motion_list[0]
            motion = motions_values[motion_idx]
            
            # print motion_list,motion
            
            digit = data[idxs]
            dummy = create_gif(digit, motion)
            
            # print dummy.shape
            
            final_gif_data[outer_index, :, :, :, :] = dummy

            sentence = 'the digits %d is moving %s .' % (labels[idxs[0]],motion_strings[motion])
            #k = randint(0)
        #     k = 0

        # #    if k ==0:
        # #        sentence = 'the digits %d and %d are bouncing here and there .'%(labels[idxs[0]], labels[idxs[1]])
        # #    elif k == 1:
        # #        sentence = 'the digits %d and %d are moving here and there .' %(labels[idxs[0]], labels[idxs[1]])
        # #    elif k == 2:
        # #        sentence = 'the digits %d and %d are jumping up and down .' %(labels[idxs[0]], labels[idxs[1]])
        # #    elif k == 3:
        # #        sentence = 'the digits %d and %d are bouncing up and down .' %(labels[idxs[0]], labels[idxs[1]])
        #     if k ==0:
        #         sentence = 'the digits %d is moving left and right .'%(labels[idxs[0]])
        #     elif k == 1:
        #         sentence = 'the digits %d is moving here and there .' %(labels[idxs[0]])
        #     elif k == 2:
        #         sentence = 'the digits %d is jumping up and down .' %(labels[idxs[0]])
        #     elif k == 3:
        #         sentence = 'the digits %d is bouncing up and down .' %(labels[idxs[0]])
            counts_of_sentences[motion_idx] += 1
            print sentence
            
            sentence_matrix = sent2matrix(sentence, dictionary)
            captions[outer_index, :] = sentence_matrix
            #break
        else: 
            outer_index -= 1
        outer_index += 1
        # if outer_index == 10:
        #     break
        
    return final_gif_data,captions,counts_of_sentences

f = h5py.File('mnist.h5')
train_data = f['train'].value.reshape(-1, 28, 28)
train_labels = f['train_labels'].value
val_data = f['validation'].value.reshape(-1,28,28)
val_labels = f['validation_labels'].value
f.close()

data = np.concatenate((train_data,val_data), axis = 0)
labels = np.concatenate((train_labels,val_labels), axis = 0)

train,val = create_dataset()
print train,val
data_train,captions_train,count_train = create_gifs_for_data(train,data,labels,10000)
data_val,captions_val,count_val = create_gifs_for_data(val,data, labels, 2000)
#data_test, captions_test, count_test = create_gifs_for_data(test,data,labels,2000)

# # np.save('video_mnist_with_captions',final_data)
# f = h5py.File('../mnist.h5')
# data = f['train'].value.reshape(-1, 28, 28)
# labels = f['train_labels'].value
# data_val = f['validation'].value.reshape(-1,28,28)
# labels_val = f['validation_labels'].value
# f.close()
# final_gif_data,captions,counts_of_sentences = create_gifs_for_data(data,labels)
with h5py.File('mnist_single_gif.h5','w') as hf:
    # final_gif_data,captions,counts_of_sentences = create_gifs_for_data(data,labels)
    hf.create_dataset('mnist_gif_train', data=data_train)
    hf.create_dataset('mnist_captions_train', data=captions_train)
    hf.create_dataset('mnist_count_train', data=count_train)
    hf.create_dataset('mnist_dataset_train', data=train)

    hf.create_dataset('mnist_gif_val', data=data_val)
    hf.create_dataset('mnist_captions_val', data=captions_val)
    hf.create_dataset('mnist_count_val', data=count_val)
    hf.create_dataset('mnist_dataset_val', data=val)

#    hf.create_dataset('mnist_gif_test', data=data_test)
#    hf.create_dataset('mnist_captions_test', data=captions_test)
#    hf.create_dataset('mnist_count_test', data=count_test)
#    hf.create_dataset('mnist_dataset_test', data=test)
    # #np.save('three_framed_dataset.npy', final_gif_data)
    # final_gif_data,captions,counts_of_sentences = create_gifs_for_data(data_val,labels_val)
    # hf.create_dataset('mnist_gif_val', data=final_gif_data)
    # hf.create_dataset('mnist_captions_val', data=captions)
    # hf.create_dataset('mnist_count_val', data=counts_of_sentences)

