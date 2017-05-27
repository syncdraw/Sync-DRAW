'''before running this code run any of the codes in dataset folder to create a .h5 file of the dataset
Once the dataset is created, give the path to the datset to the variable dataset_file'''
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import h5py
import logging

model_file_name = "results_twodigit/unsupervised_frame_cvae_10"

## MODEL PARAMETERS ## 
C = 10
A,B = 64,64 # image width,height
img_size = B*A # the canvas size
gif_size = C*B*A # gif sizee
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 10 # read glimpse grid width/height
write_n = 10 # write glimpse grid width/height
read_size = 2*read_n*read_n #if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n #if FLAGS.write_attn else img_size
z_size=100 # QSampler output size
T=10 # MNIST generation sequence length
batch_size=100 # training minibatch size
train_iters=50000
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,C,img_size)) # input (batch_size * img_size)
#e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

def next_batch(data_array):
    length=data_array.shape[0] #assuming the data array to be a np arry
    permutations=np.random.permutation(length)
    idxs=permutations[0:batch_size]
    batch=np.zeros([batch_size, gif_size], dtype=np.float32)
    for i in range(len(idxs)):
        batch[i,:]=data_array[idxs[i]].flatten()
    return batch 

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

# def filterbank(gx, gy, sigma2,delta, N):
#     grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
#     mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
#     mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
#     a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
#     b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
#     mu_x = tf.reshape(mu_x, [-1, N, 1])
#     mu_y = tf.reshape(mu_y, [-1, N, 1])
#     sigma2 = tf.reshape(sigma2, [-1, 1, 1])
#     Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
#     Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
#     # normalize, sum over A and B dims
#     Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
#     Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
#     return Fx,Fy

def filterbank_gif(gx, gy, sigma2, delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, 1, -1])

    grid_i_gif = tf.tile(grid_i,[batch_size,C,1]) # (BxCxN)
    gx = tf.reshape(gx,[batch_size,C,1]) # (BxCx1)
    gy = tf.reshape(gy,[batch_size,C,1]) # (BxCx1)
    delta = tf.reshape(delta, [batch_size,C,1])

    print grid_i_gif.get_shape(),gx.get_shape(),gy.get_shape()

    mu_x = gx + (grid_i_gif - N / 2 - 0.5) * delta # (BxCxN)
    mu_y = gy + (grid_i_gif - N / 2 - 0.5) * delta # (BxCxN)

    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, 1, -1]) # (1x1x1xA)
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, 1, -1]) # (1x1x1xB)

    a = tf.tile(a,[batch_size,C,1,1]) # (1xCx1xA)
    b = tf.tile(b,[batch_size,C,1,1]) # (1xCx1xB)

    mu_x = tf.reshape(mu_x, [-1, C, N, 1]) # (BxCxNx1)
    mu_y = tf.reshape(mu_y, [-1, C, N, 1]) # (BxCxNx1)
    sigma2 = tf.reshape(sigma2, [-1, C, 1, 1]) # (BxCx1x1)

    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # (BxCxNxA)
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # (BxCxNxB)

    Fx = Fx/tf.maximum(tf.reduce_sum(Fx,3,keep_dims=True),eps) # (BxCxNxA)
    Fy = Fy/tf.maximum(tf.reduce_sum(Fy,3,keep_dims=True),eps) # (BxCxNxB)

    return Fx,Fy


def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5*C)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    # frame_params = tf.nn.softmax(frame_params)
    # frame = tf.cast(frame,tf.int32)
    # print frame_params.get_shape()

    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x 

    return filterbank_gif(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev):
    return tf.concat(1,[x,x_hat])

def read_attn(x,x_hat,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    # def filter_img(img,Fx,Fy,gamma,N,frame):
    #     # img_list = []
    #     # for i in range(batch_size):
    #     #     img_list.append(tf.slice(gif,[i,frame[i],0],[1,1,-1]))
    #     # img = tf.pack(img_list)
    #     #not supported for gradient as of now - sad :(
    #     # indices = tf.concat(1,[tf.reshape(tf.cast(tf.range(batch_size),tf.int64),[-1,1]),tf.reshape(frame,[-1,1])])
    #     # img = tf.gather_nd(gif, indices)#tf.slice(gif,[0,frame,0],[-1,1,-1])
    #     Fxt=tf.transpose(Fx,perm=[0,2,1])
    #     img=tf.reshape(img,[-1,B,A])
    #     glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
    #     glimpse=tf.reshape(glimpse,[-1,N*N])
    #     return glimpse*tf.reshape(gamma,[-1,1])*tf.reshape(frame,[-1,1])

    def filter_gif(gif,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,1,3,2])
        gif=tf.reshape(gif,[-1,C,B,A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(gif,Fxt))
        glimpse=tf.reshape(glimpse,[-1,C,N*N])
        return tf.reshape(glimpse*tf.reshape(gamma,[-1,C,1]),[-1,C*N*N])
    # x_img_list = []
    # x_hat_img_list = []
    # for i in range(C):
    #     x_img_list.append(filter_img(tf.slice(x,[0,i,0],[-1,1,-1]),Fx[:,i,:],Fy,gamma,read_n,tf.slice(frame_params,[0,i],[-1,1])))
    #     x_hat_img_list.append(filter_img(tf.slice(x_hat,[0,i,0],[-1,1,-1]),Fx,Fy,gamma,read_n,tf.slice(frame_params,[0,i],[-1,1])))
    # x = tf.pack(x_img_list,axis=1)
    # x_hat = tf.pack(x_hat_img_list,axis=1)

#    print x.get_shape()
#    print x_hat.get_shape()
    # x=filter_img(x,Fx,Fy,gamma,read_n,frame) # batch x (read_n*read_n)
    # x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n,frame)
    x=filter_gif(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    x_hat=filter_gif(x_hat,Fx,Fy,gamma,read_n)
    return tf.concat(1,[x,x_hat]),gamma # concat along feature axis

read = read_attn #if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc,e):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(input, state)

## WRITER ## 
def write_no_attn(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,C*write_size) # batch x (write_n*write_n)
    N=write_n
    w=tf.reshape(w,[batch_size,C,N,N])
    Fx,Fy,gamma=attn_window("write",h_dec,write_n)
    Fyt=tf.transpose(Fy,perm=[0,1,3,2])
    wr=tf.batch_matmul(Fyt,tf.batch_matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,C,B*A])

    # wr_frames = tf.reshape(wr,[batch_size,1,B*A])
    
    # frames_params_arg_max = tf.argmax(frame_params,1)
    # frame_params_one_hot = tf.one_hot(frames_params_arg_max,C) 
    # frame_params_t = tf.reshape(frame_params_one_hot,[batch_size,C,1])
    # to_write = tf.batch_matmul(frame_params_t,wr_frames)
    print wr.get_shape()
    #gamma=tf.tile(gamma,[1,B*A])
    return wr*tf.reshape(1.0/gamma,[-1,C,1]),gamma


write=write_attn #if FLAGS.write_attn else write_no_attn

## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
read_frame = [0]*T
write_frame = [0]*T
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,C,img_size)) if t==0 else cs[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    r,frame=read(x,x_hat,h_dec_prev)
    read_frame[t] = frame
    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    e=tf.random_normal((batch_size,z_size), mean=0, stddev=1)
    z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc,e)
    h_dec,dec_state=decode(dec_state,z)
    glimpse_write,frame = write(h_dec)
    write_frame[t] = frame
    # c_to_write = pad_glimpse(glimpse_write,frame)
    cs[t]=c_prev+glimpse_write # store results
    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

## Testing code ##
cs_test = [0] * T
h_dec_test = [0] * T
dec_state_test = lstm_dec.zero_state(batch_size, tf.float32)

for t in range(T):
    e = tf.random_normal((batch_size, z_size), mean = 0, stddev = 1)
    c_prev_test = tf.zeros((batch_size, C, img_size) if t == 0 else cs_test[t-1])
    h_dec_test[t], dec_state_test = decoder(dec_state, e)
    glimpse_write_test, frame_test = write(h_dec_test[t])
    cs_test[t] = c_prev_test + glimpse_write_test


## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(tf.reshape(x,[batch_size,-1]),tf.reshape(x_recons,[batch_size,-1])),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

## OPTIMIZER ## 

# learning_rate = tf.train.exponential_decay(
#   0.01,                # Base learning rate.
#   train_iters,  # Current index into the dataset.
#   2500,          # Decay step.
#   0.95,                # Decay rate.
#   staircase=True)

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING ## 

# data_directory = os.path.join(FLAGS.data_dir, "MNIST_data")
# if not os.path.exists(data_directory):
#   os.makedirs(data_directory)
# train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data

# with h5py.File('/home/ee13b1044/honours/TGIF-Release-master/data/gif_data.h5','r') as hf:
#     inputImages = np.float32(np.array(hf.get('gif_data')).reshape(-1,C,4096))/255.


dataset_file = '/path/to/dataset/'
with h5py.File(dataset_file,'r') as hf:
    inputImages = np.float32(np.array(hf.get('mnist_gif_train')).reshape(-1,C,4096))
    #inputImages_val = np.float32(np.array(hf.get('mnist_gif_val')).reshape(-1,C,4096))

train_data = inputImages
#val_data = inputImages_val
# train_data = np.load('single_bouncing_mnist.npy')
print "loaded"

fetches=[]
fetches.extend([Lx,Lz,read_frame,write_frame,train_op])
Lxs=[0]*train_iters
Lzs=[0]*train_iters
rf=[0]*train_iters
wf=[0]*train_iters

sess=tf.InteractiveSession()

saver = tf.train.Saver() # saves variables learned during training


if os.path.isfile(model_file_name+".ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, model_file_name+".ckpt")
#     canvases,h_dec_ts = sess.run([cs_test,h_dec_test],feed_dict={})
#     canvases = np.array(canvases)
else:
    tf.initialize_all_variables().run()
    for i in range(train_iters):
        xtrain=next_batch(train_data) # xtrain is (batch_size x img_size)
        # xtrain = xtrain.reshape(-1,C,img_size) # xtrain is (batch_size x img_size)
        for j in range(C-1):
            if j==0:
                feed_dict={x:xtrain[:,j],x_prev:np.float32(np.zeros((batch_size,img_size)))}
            else:
                feed_dict={x:xtrain[:,j],x_prev:xtrain[:,j-1]}
            results=sess.run(fetches,feed_dict)

        # feed_dict={x:xtrain,y:ytrain}
        # results=sess.run(fetches,feed_dict)
        Lxs[i],Lzs[i],et,_=results
        if i%10==0:
            print("iter=%d : Lx: %f Lz: %f" % (i,Lxs[i],Lzs[i]))
            #print np.array(et)[:,0,:]
        if (i+1)%500==0:
            ckpt_file=model_file_name+".ckpt"
            print("Model saved in file: %s" % saver.save(sess,ckpt_file))
    print("training is finished")
    
    ## testing phase ##
    canvases, h_dec_ts=sess.run([cs_test, h_dec_test], feed_dict = {}) # generate some examples
    canvases=np.array(canvases) # T x batch x img_size
    xt = 1./(1 + np.exp(canvases))
    out_file=model_file_name+".npy"
    np.save(out_file,[xt,Lxs,Lzs])
    print("Outputs saved in file: %s" % out_file)

    ckpt_file=model_file_name+".ckpt"
    print("Model saved in file: %s" % saver.save(sess,ckpt_file))

    sess.close()

    print('Done drawing! Have a nice day! :)')
