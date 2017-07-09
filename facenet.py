# -*- coding: utf-8 -*-
import os,sys,shutil,argparse,cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v4

#########################################################
class BatchGenerator:
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.image = mnist.train.images
        self.label = mnist.train.labels

        self.image = np.reshape(self.image, [len(self.image), 28, 28])

    def getOne(self):
        idx = np.random.randint(0,len(self.image)-1)
        x,t = self.image[idx],self.label[idx]
        if color:
            x = np.expand_dims(x,axis=2)
            x = np.tile(x,(1,3))
        return x,t

    def getBatch(self,nBatch,color=True):
        idx = np.random.randint(0,len(self.image)-1,nBatch)
        x,t = self.image[idx],self.label[idx]
        if color:
            x = np.expand_dims(x,axis=3)
            x = np.tile(x,(1,1,3))
        return x,t

#########################################################
class FaceNet:
    def __init__(self,isTraining,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.pretrainedModelPath = "inception_v4.ckpt"
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = args.imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.alpha = args.alpha
        self.imgSize = [224,224,3]
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def getShape(self,h):
        return [int(x) for x in h.get_shape()]


    def buildNN(self,x,reuse=False,isTraining=True):
        h = x

        # Slim
        _,end_points = inception_v4.inception_v4(h,is_training=isTraining,reuse=reuse)
        h = end_points["Mixed_5b"]
        if not reuse:
            self.variables_to_restore_bbNet = slim.get_variables_to_restore()

        with tf.variable_scope("NN") as scope:
            if reuse: scope.reuse_variables()

            # avgpool
            n_b, n_h, n_w, n_f = self.getShape(h)
            h = tf.nn.avg_pool(h,ksize=[1,n_h,n_w,1],strides=[1,1,1,1],padding="VALID")

            # fc1
            n_b, n_h, n_w, n_f = self.getShape(h)
            assert n_h==n_w==1, "invalid shape after avg pool:(%d,%d)"%(n_h,n_w)
            h = tf.reshape(h,[n_b,n_f])
            self.fc1_w, self.fc1_b = self._fc_variable([n_f,128],name="fc1")
            h = tf.matmul(h, self.fc1_w) + self.fc1_b

            # L2 normalization
            h = h / tf.norm(h,axis=1,keep_dims=True)

        return h

    def buildModel(self):
        # define variables
        self.t1 = tf.placeholder(tf.int32  , [self.nBatch],name="cls1")
        self.t2 = tf.placeholder(tf.int32  , [self.nBatch],name="cls2")
        self.x1 = tf.placeholder(tf.float32, [self.nBatch, self.imgSize[0], self.imgSize[1], self.imgSize[2]],name="x1")
        self.x2 = tf.placeholder(tf.float32, [self.nBatch, self.imgSize[0], self.imgSize[1], self.imgSize[2]],name="x2")
        self.y1 = self.buildNN(self.x1,reuse=False)
        self.y2 = self.buildNN(self.x2,reuse=True)

        # define loss
        isSameClass = tf.cast(tf.equal(self.t1,self.t2),tf.int32)
        distance = tf.norm(self.y1-self.y2,axis=1)
        self.L_pos = distance
        self.L_neg = tf.maximum(self.alpha - distance,0) # if the condition is satisfied, then neglect
        self.lossElement = isSameClass * self.L_pos + (1-isSameClass) * self.L_neg
        self.loss  = tf.reduce_mean(self.lossElement)

        # define optimizer
        varList = None
        if not self.doFineTune:
            varList = [x for x in tf.trainable_variables() if "NN" in x.name]
        self.optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss, var_list=varList)

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.90))
        self.sess = tf.Session(config=config)
        #############################
        ### saver
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary = tf.summary.merge_all()
        if self.saveFolder:
            if not os.path.exists(self.saveFolder):
                os.makedirs(self.saveFolder)
                self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)
                #self.writer_train = tf.summary.FileWriter(os.path.join(self.saveFolder,"train"), self.sess.graph)
                #self.writer_test  = tf.summary.FileWriter(os.path.join(self.saveFolder,"test"))
        return

    def loadModel(self, model_path=None, pre_trained_path=None):
        if model_path:
            self.saver.restore(self.sess, model_path)
        else:
            # model_pathが設定されて居ない場合でも、pre-trainedなモデルだけは読み込む
            self.saver_pretrained = tf.train.Saver(self.variables_to_restore)
            self.saver_pretrained.restore(self.sess, pre_trained_path)


    def train(self,f_batch):
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15))
        self.sess = tf.Session(config=config)

        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        epoch = -1
        d_loss, g_loss = 1., 1.
        d_accuracy, g_accuracy = -1., -1.
        while True:
            epoch += 1

            batch_images,_ = f_batch(self.nBatch)
            batch_z        = np.random.uniform(-1,+1,[self.nBatch,self.zdim]).astype(np.float32)

            # update generator
            if d_loss > g_loss:
                _,d_loss,d_accuracy = self.sess.run([self.d_optimizer,self.d_loss,self.d_accuracy],feed_dict={self.z:batch_z, self.y_real:batch_images})
            else:
                _,g_loss,g_accuracy = self.sess.run([self.g_optimizer,self.g_loss,self.g_accuracy],feed_dict={self.z:batch_z, self.y_real:batch_images})
            if epoch%1000==0:
                print "%4d: loss(discri)=%.2e, loss(gener)=%.2e, accuracy(discri)=%.1f%%, accuracy(gener)=%.1f%%"%(epoch,d_loss,g_loss,d_accuracy*100., g_accuracy*100.)
                g_image = self.sess.run(self.y_sample,feed_dict={self.z:np.random.uniform(-1,+1,[self.nBatch,self.zdim]).astype(np.float32)})
                cv2.imwrite("log/img_%d.png"%epoch,g_image[0]*255.)
                #cv2.imwrite("log/img_%d.png"%epoch,batch_images[0]*255.)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch","-b",dest="nBatch",type=int,default=16)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-3)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="models")
    parser.add_argument("--reload","-l",dest="reload",type=str,default=None)
    parser.add_argument("--DBpath","-d",dest="DBpath",type=str,default="/home/ysasaki/data/Yasukawa/_Engine2_2017.05.25.db")
    parser.add_argument("--NNmode","-m",dest="NNmode",type=str,choices=["actor","critic","distal","Qcritic"],default="distal")
    parser.add_argument("--zdim","-z",dest="zdim",type=int,default=128)
    parser.add_argument("--alpha","-a",dest="alpha",type=float,default=0.2)
    args = parser.parse_args()
    args.imageSize = [224,224,3]
    net = FaceNet(isTraining=True,args=args)
    net.train()
