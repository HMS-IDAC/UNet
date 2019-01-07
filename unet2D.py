import numpy as np
from scipy import misc
import tensorflow as tf
import shutil
import scipy.io as sio
import os

# --------------------------------------------------
# setup
# --------------------------------------------------

imsize = 360
nclasses = 3
nchannels = 1
ntrain = 60
nvalid = 10
ntest = 23
nsteps = 10#000
batchsize = 4
nextraconvs = 1

restore_variables = False

ks = 3
nout0 = 16
nout1 = 2*nout0
nout2 = 2*nout1
nout3 = 2*nout2
dsf0 = 2 # downsampling factor, first layer
dsf1 = 2 # downsampling factor, second layer
dsf2 = 2 # downsampling factor, third layer


# noutX = [nchannels, nout0, nout1]
# dsfX = [dsf0]

# noutX = [nchannels, nout0, nout1, nout2]
# dsfX = [dsf0, dsf1]

noutX = [nchannels, nout0, nout1, nout2, nout3]
dsfX = [dsf0, dsf1, dsf2]

nlayers = len(dsfX)


impath = '/home/mc457/Workspace/DH_NBCB'

train_writer_path = '/home/mc457/Workspace/TFLog/UNet/Train'
valid_writer_path = '/home/mc457/Workspace/TFLog/UNet/Valid'
out_log_path = '/home/mc457/Workspace/TFLog/UNet'
out_model_path = '/home/mc457/Workspace/TFModel/UNet.ckpt'
out_pm_path = '/home/mc457/Workspace/Scratch'


def concat3(lst):
    return tf.concat(3,lst)

# --------------------------------------------------
# data
# --------------------------------------------------

Train = np.zeros((ntrain,imsize,imsize,nchannels))
Valid = np.zeros((nvalid,imsize,imsize,nchannels))
Test = np.zeros((ntest,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain,imsize,imsize,nclasses))
LValid = np.zeros((nvalid,imsize,imsize,nclasses))
LTest = np.zeros((ntest,imsize,imsize,nclasses))

for isample in range(0, ntrain):
    path = '%s/TrainImages/I%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)/255
    im = im-np.mean(im)
    im = im/np.std(im)
    Train[isample,:,:,0] = im
    path = '%s/TrainLabels/L%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)
    for i in range(nclasses):
        if nclasses == 2:
            LTrain[isample,:,:,i] = (im == 2*i)
        else:
            LTrain[isample,:,:,i] = (im == i)

for isample in range(0, nvalid):
    path = '%s/ValidImages/I%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)/255
    im = im-np.mean(im)
    im = im/np.std(im)
    Valid[isample,:,:,0] = im
    path = '%s/ValidLabels/L%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)
    for i in range(nclasses):
        if nclasses == 2:
            LValid[isample,:,:,i] = (im == 2*i)
        else:
            LValid[isample,:,:,i] = (im == i)

for isample in range(0, ntest):
    path = '%s/TestImages/I%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)/255
    im = im-np.mean(im)
    im = im/np.std(im)
    Test[isample,:,:,0] = im
    path = '%s/TestLabels/L%03d.tif' % (impath,isample+1)
    im = misc.imread(path).astype(np.float32)
    for i in range(nclasses):
        if nclasses == 2:
            LTest[isample,:,:,i] = (im == 2*i)
        else:
            LTest[isample,:,:,i] = (im == i)

# --------------------------------------------------
# downsampling layer
# --------------------------------------------------

with tf.name_scope('placeholders'):
    tf_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels],name='data')
    tf_labels = tf.placeholder("float", shape=[None,imsize,imsize,nclasses],name='labels')

def down_samp_layer(data,index):
    with tf.name_scope('ld%d' % index):
        ldX_weights1 = tf.Variable(tf.truncated_normal([ks, ks, noutX[index], noutX[index+1]], stddev=0.1),name='kernel1')
        ldX_weights_extra = []
        for i in range(nextraconvs):
            ldX_weights_extra.append(tf.Variable(tf.truncated_normal([ks, ks, noutX[index+1], noutX[index+1]], stddev=0.1),name='kernel_extra%d' % i))
        
        c_00 = tf.nn.relu(tf.nn.conv2d(data, ldX_weights1, strides=[1, 1, 1, 1], padding='SAME'),name='conv')
        for i in range(nextraconvs):
            c_00 = tf.nn.relu(tf.nn.conv2d(c_00, ldX_weights_extra[i], strides=[1, 1, 1, 1], padding='SAME'),name='conv_extra%d' % i)
        return tf.nn.max_pool(c_00, ksize=[1, dsfX[index], dsfX[index], 1], strides=[1, dsfX[index], dsfX[index], 1], padding='SAME',name='maxpool')

# --------------------------------------------------
# bottom layer
# --------------------------------------------------

with tf.name_scope('lb'):
    lb_weights1 = tf.Variable(tf.truncated_normal([ks, ks, noutX[nlayers], noutX[nlayers+1]], stddev=0.1),name='kernel1')
    def lb(hidden):
        return tf.nn.relu(tf.nn.conv2d(hidden, lb_weights1, strides=[1, 1, 1, 1], padding='SAME'),name='conv')

# --------------------------------------------------
# downsampling
# --------------------------------------------------

with tf.name_scope('downsampling'):    
    dsX = []
    dsX.append(tf_data)

    for i in range(nlayers):
        dsX.append(down_samp_layer(dsX[i],i))

    b = lb(dsX[nlayers])

# --------------------------------------------------
# upsampling layer
# --------------------------------------------------

def up_samp_layer(data,index):
    with tf.name_scope('lu%d' % index):
        luX_weights1    = tf.Variable(tf.truncated_normal([ks, ks, noutX[index+1], noutX[index+2]], stddev=0.1),name='kernel1')
        luX_weights2    = tf.Variable(tf.truncated_normal([ks, ks, noutX[index]+noutX[index+1], noutX[index+1]], stddev=0.1),name='kernel2')
        luX_weights_extra = []
        for i in range(nextraconvs):
            luX_weights_extra.append(tf.Variable(tf.truncated_normal([ks, ks, noutX[index+1], noutX[index+1]], stddev=0.1),name='kernel2_extra%d' % i))
        
        out_size = imsize
        for i in range(index):
            out_size /= dsfX[i]

        output_shape = [batchsize,out_size,out_size,noutX[index+1]]
        us = tf.nn.relu(tf.nn.conv2d_transpose(data, luX_weights1, output_shape, strides=[1, dsfX[index], dsfX[index], 1], padding='SAME'),name='conv1')
        cc = concat3([dsX[index],us]) 
        cv = tf.nn.relu(tf.nn.conv2d(cc, luX_weights2, strides=[1, 1, 1, 1], padding='SAME'),name='conv2')
        for i in range(nextraconvs):
            cv = tf.nn.relu(tf.nn.conv2d(cv, luX_weights_extra[i], strides=[1, 1, 1, 1], padding='SAME'),name='conv2_extra%d' % i)
        return cv

# --------------------------------------------------
# final (top) layer
# --------------------------------------------------

with tf.name_scope('lt'):
    lt_weights1    = tf.Variable(tf.truncated_normal([1, 1, noutX[1], nclasses], stddev=0.1),name='kernel')
    def lt(hidden):
        return tf.nn.conv2d(hidden, lt_weights1, strides=[1, 1, 1, 1], padding='SAME',name='conv')


# --------------------------------------------------
# upsampling
# --------------------------------------------------

with tf.name_scope('upsampling'):
    usX = []
    usX.append(b)

    for i in range(nlayers):
        usX.append(up_samp_layer(usX[i],nlayers-1-i))

    t = lt(usX[nlayers])

# --------------------------------------------------
# optimization
# --------------------------------------------------

with tf.name_scope('optim'):
    sm = tf.nn.softmax(t,-1)
    loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf_labels,tf.log(sm)),3))
    opt_op = tf.train.MomentumOptimizer(1e-3,0.9).minimize(loss)
    # opt_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

# error
with tf.name_scope('eval'):
    error = []
    for iclass in range(nclasses):
        labels0 = tf.reshape(tf.to_int32(tf.slice(tf_labels,[0,0,0,iclass],[-1,-1,-1,1])),[batchsize,imsize,imsize])
        predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(sm,3),iclass)),[batchsize,imsize,imsize])
        correct = tf.multiply(labels0,predict0)
        ncorrect0 = tf.reduce_sum(correct)
        nlabels0 = tf.reduce_sum(labels0)
        error.append(1-tf.to_float(ncorrect0)/tf.to_float(nlabels0))
    errors = tf.tuple(error)

# --------------------------------------------------
# inspection
# --------------------------------------------------

with tf.name_scope('loss'):
    tf.summary.scalar('avg_cross_entropy', loss)
    for iclass in range(nclasses):
        tf.summary.scalar('avg_pixel_error_%d' % iclass, error[iclass])
with tf.name_scope('histograms'):
    tf.summary.histogram('ds0',dsX[1])
with tf.name_scope('images'):
    split0 = tf.slice(sm,[0,0,0,0],[-1,-1,-1,1])
    split1 = tf.slice(sm,[0,0,0,1],[-1,-1,-1,1])
    tf.summary.image('pm0',split0)
    tf.summary.image('pm1',split1)
merged = tf.summary.merge_all()


# --------------------------------------------------
# session
# --------------------------------------------------

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

if os.path.exists(out_log_path):
    shutil.rmtree(out_log_path)
train_writer = tf.summary.FileWriter(train_writer_path, sess.graph)
valid_writer = tf.summary.FileWriter(valid_writer_path, sess.graph)

if restore_variables:
    saver.restore(sess, out_model_path)
    print("Model restored.")
else:
    sess.run(tf.global_variables_initializer())

# --------------------------------------------------
# train
# --------------------------------------------------

batch_data = np.zeros((batchsize,imsize,imsize,nchannels))
batch_labels = np.zeros((batchsize,imsize,imsize,nclasses))
for i in range(nsteps):
    # train

    perm = np.arange(ntrain)
    np.random.shuffle(perm)

    for j in range(batchsize):
        batch_data[j,:,:,:] = Train[perm[j],:,:,:]
        batch_labels[j,:,:,:] = LTrain[perm[j],:,:,:]

    summary,_ = sess.run([merged,opt_op],feed_dict={tf_data: batch_data, tf_labels: batch_labels})
    train_writer.add_summary(summary, i)

    # validation

    perm = np.arange(nvalid)
    np.random.shuffle(perm)

    for j in range(batchsize):
        batch_data[j,:,:,:] = Valid[perm[j],:,:,:]
        batch_labels[j,:,:,:] = LValid[perm[j],:,:,:]

    summary, es = sess.run([merged, errors],feed_dict={tf_data: batch_data, tf_labels: batch_labels})
    valid_writer.add_summary(summary, i)

    e = np.mean(es)
    print('step %05d, e: %f' % (i,e))

    if i == 0:
        if restore_variables:
            lowest_error = e
        else:
            lowest_error = np.inf

    if np.mod(i,100) == 0 and e < lowest_error:
        lowest_error = e
        print("Model saved in file: %s" % saver.save(sess, out_model_path))


# --------------------------------------------------
# test
# --------------------------------------------------

if not os.path.exists(out_pm_path):
    os.makedirs(out_pm_path)

for i in range(ntest):
    j = np.mod(i,batchsize)

    batch_data[j,:,:,:] = Test[i,:,:,:]
    batch_labels[j,:,:,:] = LTest[i,:,:,:]
 
    if j == batchsize-1 or i == ntest-1:

        output = sess.run(sm,feed_dict={tf_data: batch_data, tf_labels: batch_labels})

        for k in range(j+1):
            sio.savemat('%s/PM_I%02d.mat' % (out_pm_path,i-j+k+1), {'classProbs':output[k,:,:,:]})
            for l in range(nclasses):
                misc.imsave('%s/I%d_PM%d.png' % (out_pm_path,i-j+k+1,l),output[k,:,:,l])

# --------------------------------------------------
# clean-up
# --------------------------------------------------

train_writer.close()
valid_writer.close()
sess.close()
