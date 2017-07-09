# -*- coding: utf-8 -*-
import os,sys,shutil,argparse,cv2,glob,csv,random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v4

#########################################################
class BatchGenerator:
    # assumes name/filename.jpg
    def __init__(self,zdim,imageShape=[224,224,3]):
        self.imageShape = imageShape
        self.zdim = zdim

    def loadAndSaveDir(self,path,outFile="class.csv",trainFrac=0.9):
        self.data = {}
        count_keys = 0
        count_file = 0
        for f in glob.glob(os.path.join(path,"*/*.jpg")):
            #print f
            fullPath  = f
            className = f.split("/")[-2]
            if not className in self.data:
                self.data[className] = []
                count_keys += 1
            self.data[className].append(fullPath)
            count_file += 1

        print "found %d files, %d unique classes"%(count_file, count_keys)

        self.keys_train = random.sample(self.data.keys(),int(len(self.data.keys())*trainFrac))
        self.keys_test  = list( set(self.data.keys()) - set(self.keys_train) )

        with open(outFile,"w") as ofile:
            w = csv.writer(ofile,lineterminator="\n")
            for cls in self.keys_train:
                w.writerow([cls,"train"])
            for cls in self.keys_test:
                w.writerow([cls,"test"])

        self.keys = {}
        self.keys["train"] = self.keys_train
        self.keys["test"]  = self.keys_test

        def makeArray(tgtDict):
            clsIndex = []
            imgPath  = []
            cls_index = -1
            mycount = 0
            for cls in tgtDict:
                cls_index += 1
                for f in self.data[cls]:
                    mycount += 1
                    clsIndex.append(cls_index)
                    imgPath.append (f)
            clsIndex = np.array(clsIndex)
            imgPath  = np.array(imgPath)
            tempVec  = np.zeros([mycount,self.zdim],dtype=np.float32)

            return clsIndex,imgPath,tempVec

        self.clsIndex,self.imgPath,self.tempVec = {},{},{}
        self.clsIndex["train"], self.imgPath["train"], self.tempVec["train"] = makeArray(self.keys_train)
        self.clsIndex["test"] , self.imgPath["test"] , self.tempVec["test"]  = makeArray(self.keys_test )

        return

    def setVec(self,v1,v2):
        self.tempVec[self.prevMode][self.prevIndex[:,0]] = v1
        self.tempVec[self.prevMode][self.prevIndex[:,1]] = v2
        return

    def getBatch(self,nBatch,alpha,mode="train",nOneTry=20):
        def oneTry():
            idx1 = np.random.randint(0,self.tempVec[mode].shape[0]-1,nOneTry)
            idx2 = np.random.randint(0,self.tempVec[mode].shape[0]-1,nOneTry)
            x1   = self.tempVec[mode][idx1]
            x2   = self.tempVec[mode][idx2]
            # 同じクラスになっているものはベクトルの長さを0としてしまう→後でalphaより小さいか否かを確認する際に必ず選定されることになる
            same = np.equal(self.clsIndex[mode][idx1],self.clsIndex[mode][idx2])
            x1[same,:] = 0.
            x2[same,:] = 0.
            # ベクトル間の距離がalpha以下ならば、それが選定される
            x1 = np.repeat(x1,nOneTry,axis=0)
            x2 = np.tile  (x2,(nOneTry,1))
            dist = np.linalg.norm(x1-x2,axis=1)
            good = np.where(dist<alpha)
            i1 = np.divide(good,nOneTry)
            i2 = good - i1*nOneTry
            return idx1[i1][0],idx2[i2][0]

        # 最低nBatch個の組を作成 
        self.prevIndex = idx = np.zeros([nBatch,2],np.int32)
        self.prevMode  = mode
        cnt = 0
        while True:
            idList1,idList2 = oneTry()
            #for k in range(res[0].shape[0]):
            for id1,id2 in zip(idList1,idList2):
                idx[cnt,0] = id1
                idx[cnt,1] = id2
                cnt += 1
                if cnt == nBatch: break
            if cnt == nBatch: break

        # nBatch個に対し、ファイルの読み込みを実施
        # ToDo: LFWデータセットは縦横同じ長さなので問題ないが、今後他のデータセットへ拡張する際は、縦横比を崩さないよう工夫をすること
        def loadImg(url):
            img = cv2.imread(url)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(self.imageShape[0],self.imageShape[1]),cv2.INTER_CUBIC)
            return img

        imgList1 = np.zeros([nBatch, self.imageShape[0], self.imageShape[1], self.imageShape[2]], dtype=np.float32)
        imgList2 = np.zeros([nBatch, self.imageShape[0], self.imageShape[1], self.imageShape[2]], dtype=np.float32)
        clsList1 = np.zeros([nBatch], dtype=np.int32)
        clsList2 = np.zeros([nBatch], dtype=np.int32)
        for i in range(nBatch):
            imgList1[i] = loadImg(self.imgPath[mode][idx[i,0]])
            imgList2[i] = loadImg(self.imgPath[mode][idx[i,1]])
            clsList1[i] = self.clsIndex[mode][idx[i,0]]
            clsList2[i] = self.clsIndex[mode][idx[i,1]]

        return imgList1,imgList2,clsList1,clsList2

#########################################################
class FaceNet:
    def __init__(self,isTraining,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.pretrained_path = "./inception_v4.ckpt"
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = args.imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.alpha = args.alpha
        self.imgSize = args.imageSize
        self.doFineTune = args.doFineTune
        self.nOneTry = args.nOneTry
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
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            _,end_points = inception_v4.inception_v4(h,is_training=isTraining,reuse=reuse,create_aux_logits=False)
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
            self.fc1_w, self.fc1_b = self._fc_variable([n_f,self.zdim],name="fc1")
            h = tf.matmul(h, self.fc1_w) + self.fc1_b

            # L2 normalization
            h = h / tf.norm(h,axis=1,keep_dims=True)

        tf.summary.histogram("fc1_w"   ,self.fc1_w)
        tf.summary.histogram("fc1_b"   ,self.fc1_b)


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
        isSameClass = tf.cast(tf.equal(self.t1,self.t2),tf.float32)
        distance = tf.norm(self.y1-self.y2,axis=1)
        self.L_pos = distance
        self.L_neg = tf.maximum(self.alpha - distance,0) # if the condition is satisfied, then neglect
        self.lossElement = isSameClass * self.L_pos + (1.-isSameClass) * self.L_neg
        self.loss  = tf.reduce_mean(self.lossElement)

        # define optimizer
        varList = None
        if not self.doFineTune:
            varList = [x for x in tf.trainable_variables() if "NN" in x.name]
        self.optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss, var_list=varList)

        tf.summary.scalar("loss" ,self.loss )

        #############################
        # define session
        #config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        #############################
        ### saver
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary = tf.summary.merge_all()
        if self.saveFolder:
            if not os.path.exists(self.saveFolder):
                os.makedirs(self.saveFolder)
            self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)
        return

    def loadModel(self, model_path=None, pretrained_path=None):
        if model_path:
            print "restoring model..."
            self.saver.restore(self.sess, model_path)
            print "done"
        else:
            # model_pathが設定されて居ない場合でも、pre-trainedなモデルだけは読み込む
            print "loading pretrained model..."
            self.saver_pretrained = tf.train.Saver(self.variables_to_restore_bbNet)
            self.saver_pretrained.restore(self.sess, pretrained_path)
            print "done"


    def train(self,bGen):
        self.loadModel(self.reload,self.pretrained_path)

        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        epoch = -1
        while True:
            epoch += 1

            x1,x2,t1,t2 = bGen.getBatch(self.nBatch,alpha=self.alpha,mode="train",nOneTry=self.nOneTry)

            # update generator
            _,loss,y1,y2,summary = self.sess.run([self.optimizer,self.loss,self.y1,self.y2,self.summary],feed_dict={self.x1:x1, self.x2:x2, self.t1:t1, self.t2:t2})
            bGen.setVec(y1,y2)
            print loss
            if epoch%100==0:
                self.writer.add_summary(summary,epoch)
            if epoch%100==0:
                print "%4d: loss=%.2e"%(epoch,loss)
            if epoch%1000==0:
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),epoch)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch","-b",dest="nBatch",type=int,default=16)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="autosave")
    parser.add_argument("--reload","-l",dest="reload",type=str,default=None)
    parser.add_argument("--DBpath","-d",dest="DBpath",type=str,default="/home/ysasaki/data/Yasukawa/_Engine2_2017.05.25.db")
    parser.add_argument("--NNmode","-m",dest="NNmode",type=str,choices=["actor","critic","distal","Qcritic"],default="distal")
    parser.add_argument("--zdim","-z",dest="zdim",type=int,default=10)
    parser.add_argument("--nOneTry","-n",dest="nOneTry",type=int,default=10)
    parser.add_argument("--alpha","-a",dest="alpha",type=float,default=0.2)
    parser.add_argument("--doFineTune","-f",dest="doFineTune",type=bool,default=False)
    args = parser.parse_args()
    args.imageSize = [112,112,3]

    bGen = BatchGenerator(zdim=args.zdim,imageShape=args.imageSize)
    bGen.loadAndSaveDir("/media/ysasaki/ForShare/data/lfw_funneled",outFile="class.csv",trainFrac=0.9)
    net = FaceNet(isTraining=True,args=args)
    net.train(bGen)
