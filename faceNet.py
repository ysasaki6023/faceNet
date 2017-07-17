# -*- coding: utf-8 -*-
import os,sys,shutil,argparse,cv2,glob,csv,random
from collections import deque
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v4

#########################################################
class BatchGenerator:
    # assumes name/filename.jpg
    def __init__(self,zdim,imageShape=[224,224,3]):
        self.imageShape = imageShape
        self.zdim = zdim

    def loadFromFile(self,path,inFile="class.csv"):
        self.keys = {}
        self.keys["train"] = []
        self.keys["test"]  = []

        with open(inFile,"r") as ofile:
            r = csv.reader(ofile)
            for line in r:
                assert line[1] in ["train","test"]
                self.keys[line[1]].append(line[0])

        self.data = {}
        count_keys = 0
        count_file = 0
        for f in glob.glob(os.path.join(path,"*/*.jpg")):
            fullPath  = f
            className = f.split("/")[-2]
            if not className in self.data:
                self.data[className] = []
                count_keys += 1
            self.data[className].append(fullPath)
            count_file += 1

        print "found %d files, %d unique classes"%(count_file, count_keys)

        self.clsIndex,self.imgPath,self.tempVec = {},{},{}
        self.clsIndex["train"], self.imgPath["train"], self.tempVec["train"] = self.makeArray(self.data,self.keys["train"])
        self.clsIndex["test"] , self.imgPath["test"] , self.tempVec["test"]  = self.makeArray(self.data,self.keys["test"] )

        return

    def loadAndSaveDir(self,path,outFile="class.csv",trainFrac=0.9):
        if not os.path.exists(os.path.dirname(outFile)):
            os.makedirs(os.path.dirname(outFile))
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

        self.clsIndex,self.imgPath,self.tempVec = {},{},{}
        self.clsIndex["train"], self.imgPath["train"], self.tempVec["train"] = self.makeArray(self.data,self.keys_train)
        self.clsIndex["test"] , self.imgPath["test"] , self.tempVec["test"]  = self.makeArray(self.data,self.keys_test )

        return

    def makeArray(self,data,tgtDict):
        clsIndex = []
        imgPath  = []
        cls_index = -1
        mycount = 0
        for cls in tgtDict:
            cls_index += 1
            for f in data[cls]:
                mycount += 1
                clsIndex.append(cls_index)
                imgPath.append (f)
        clsIndex = np.array(clsIndex)
        imgPath  = np.array(imgPath)
        tempVec  = np.zeros([mycount,self.zdim],dtype=np.float32)

        return clsIndex,imgPath,tempVec

    def setVec(self,v1,v2):
        self.tempVec[self.prevMode][self.prevIndex[:,0]] = v1
        self.tempVec[self.prevMode][self.prevIndex[:,1]] = v2
        return

    def loadImg(self,url):
        img = cv2.imread(url)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.imageShape[0],self.imageShape[1]),0,0,cv2.INTER_CUBIC)
        return img

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
            good = np.where(dist<alpha)[0]
            i1 = np.divide(good,nOneTry)
            i2 = good - i1*nOneTry
            if good.shape[0]==0: return idx1[0:1],idx2[0:1]
            myRand = np.random.randint(0,good.shape[0],good.shape[0])
            return idx1[i1][myRand],idx2[i2][myRand]

        # 最低nBatch個の組を作成 
        self.prevIndex = idx = np.zeros([nBatch,2],np.int32)
        self.prevMode  = mode
        cnt = 0
        while True:
            idList1,idList2 = oneTry()
            for id1,id2 in zip(idList1,idList2):
                idx[cnt,0] = id1
                idx[cnt,1] = id2
                cnt += 1
                if cnt == nBatch: break
            if cnt == nBatch: break

        # nBatch個に対し、ファイルの読み込みを実施
        # ToDo: LFWデータセットは縦横同じ長さなので問題ないが、今後他のデータセットへ拡張する際は、縦横比を崩さないよう工夫をすること

        pthList1 = []
        pthList2 = []
        imgList1 = np.zeros([nBatch, self.imageShape[0], self.imageShape[1], self.imageShape[2]], dtype=np.float32)
        imgList2 = np.zeros([nBatch, self.imageShape[0], self.imageShape[1], self.imageShape[2]], dtype=np.float32)
        clsList1 = np.zeros([nBatch], dtype=np.int32)
        clsList2 = np.zeros([nBatch], dtype=np.int32)
        for i in range(nBatch):
            pthList1.append(self.imgPath[mode][idx[i,0]])
            pthList2.append(self.imgPath[mode][idx[i,1]])
            imgList1[i] = self.loadImg(self.imgPath[mode][idx[i,0]])
            imgList2[i] = self.loadImg(self.imgPath[mode][idx[i,1]])
            clsList1[i] = self.clsIndex[mode][idx[i,0]]
            clsList2[i] = self.clsIndex[mode][idx[i,1]]

        return imgList1,imgList2,clsList1,clsList2,pthList1,pthList2

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

        if not reuse:
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
        isSameClassBool = tf.equal(self.t1,self.t2)
        isSameClass     = tf.cast(isSameClassBool,tf.float32)
        self.sameFrac   = tf.reduce_mean(isSameClass)
        #self.minNorm = tf.reduce_min(tf.concat([tf.norm(self.y1,axis=1),tf.norm(self.y2,axis=1)]))
        distance = tf.norm(self.y1-self.y2,axis=1)
        self.L_pos = tf.maximum(distance - self.alpha,0) # updated loss
        self.L_neg = tf.maximum(self.alpha - distance,0) # if the condition is satisfied, then neglect
        self.lossElement = isSameClass * self.L_pos + (1.-isSameClass) * self.L_neg
        self.loss  = tf.reduce_mean(self.lossElement)

        #self.count = tf.where(isSameClassBool, tf.less(distance,self.alpha))
        self.count_TT = tf.reduce_sum(tf.cast(tf.logical_and(               isSameClassBool,                tf.less(distance,self.alpha) ),tf.int32))
        self.count_TF = tf.reduce_sum(tf.cast(tf.logical_and(               isSameClassBool ,tf.logical_not(tf.less(distance,self.alpha))),tf.int32))
        self.count_FT = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(isSameClassBool),               tf.less(distance,self.alpha) ),tf.int32))
        self.count_FF = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(isSameClassBool),tf.logical_not(tf.less(distance,self.alpha))),tf.int32))

        # define optimizer
        varList = None
        if not self.doFineTune:
            varList = [x for x in tf.trainable_variables() if "NN" in x.name]
        self.optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss, var_list=varList)

        tf.summary.scalar("loss" ,self.loss )

        #############################
        # define session
        #config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.40))
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        #############################
        ### saver
        self.saver = tf.train.Saver(max_to_keep=5)
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
        count_TT, count_TF, count_FT, count_FF = deque(maxlen=1000), deque(maxlen=1000), deque(maxlen=1000), deque(maxlen=1000)
        while True:
            epoch += 1

            x1,x2,t1,t2,_,_ = bGen.getBatch(self.nBatch,alpha=self.alpha,mode="train",nOneTry=self.nOneTry)

            # update generator
            _,loss,sameFrac,y1,y2,summary,c_TT,c_TF,c_FT,c_FF = self.sess.run([self.optimizer,self.loss,self.sameFrac,self.y1,self.y2,self.summary,
                                                                               self.count_TT,self.count_TF,self.count_FT,self.count_FF],
                                                                               feed_dict={self.x1:x1, self.x2:x2, self.t1:t1, self.t2:t2})
            count_TT.append(c_TT),count_TF.append(c_TF),count_FT.append(c_FT),count_FF.append(c_FF)
            tot = float(sum(count_TT) + sum(count_TF) + sum(count_FT) + sum(count_FF))
            f_TT, f_TF, f_FT, f_FF = sum(count_TT)/tot, sum(count_TF)/tot, sum(count_FT)/tot, sum(count_FF)/tot
            bGen.setVec(y1,y2)
            if epoch%100==0:
                self.writer.add_summary(summary,epoch)
            if epoch%100==0:
                print "%6d: loss=%.2e, sameFrac=%4.1f%%, count = [[%.2f,%.2f],[%.2f,%.2f]]"%(epoch,loss,sameFrac*100.,f_TT,f_TF,f_FT,f_FF)
            if epoch%10000==0:
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),epoch)

    def test(self,outputFile):
        nMaxPerClass = 5
        self.loadModel(self.reload,self.pretrained_path)
        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        ofile = open(outputFile,"w")
        cfile = csv.writer(ofile,lineterminator="\n")

        bGenKeys = bGen.keys["train"]

        for i1 in bGenKeys:
            equal = []
            if len(bGen.data[i1])<=1: continue
            pairs = []
            while True:
                k1,k2 = np.random.randint(0,len(bGen.data[i1])),np.random.randint(0,len(bGen.data[i1]))
                if k1==k2: continue
                if k1>k2:
                    temp = k1
                    k1=k2
                    k2 = temp
                kk = (k1,k2)
                if kk in pairs: continue
                pairs.append(kk)
                equal.append(True)
                if len(pairs)>=sp.misc.comb(len(bGen.data[i1]),2,exact=True): break
                if len(pairs)>=nMaxPerClass: break
            urlList = [(bGen.data[i1][k1],bGen.data[i1][k2]) for k1,k2 in pairs]
            nSame   = len(urlList)

            for _ in range(nSame):
                i2 = random.choice(bGenKeys)
                if i2==i1: continue
                k1 = random.choice(bGen.data[i1])
                k2 = random.choice(bGen.data[i2])
                urlList.append((k1,k2))
                equal.append(False)

            for eq, url in zip(equal,urlList):
                k1,k2 = url
                x1,x2 = bGen.loadImg(k1),bGen.loadImg(k2)
                x1,x2 = np.expand_dims(x1,axis=0),np.expand_dims(x2,axis=0)
                y1,y2 = self.sess.run([self.y1,self.y2], feed_dict={self.x1:x1, self.x2:x2})
                distance = np.linalg.norm(y1-y2,axis=1)
                print eq,distance,k1,k2
                cfile.writerow([eq,distance,k1[0],k2[0],y1[0],y2[0]])
                ofile.flush()
        ofile.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch","-b",dest="nBatch",type=int,default=128)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="autosave")
    parser.add_argument("--reload","-l",dest="reload",type=str,default=None)
    parser.add_argument("--zdim","-z",dest="zdim",type=int,default=10)
    parser.add_argument("--nOneTry","-n",dest="nOneTry",type=int,default=10)
    parser.add_argument("--alpha","-a",dest="alpha",type=float,default=0.2)
    parser.add_argument("--doFineTune","-f",dest="doFineTune",action="store_true")
    parser.add_argument("--testMode","-t",dest="testMode",action="store_true")
    args = parser.parse_args()
    args.imageSize = [120,120,3]

    dataDir = "data/lfw_funneled"
    bGen = BatchGenerator(zdim=args.zdim,imageShape=args.imageSize)
    if args.testMode:
        bGen.loadFromFile(dataDir,inFile=os.path.join(os.path.dirname(args.reload),"class.csv"))
        args.nBatch = 1
        net = FaceNet(isTraining=False,args=args)
        net.test(outputFile="test.csv")
    else:
        if args.reload:
            bGen.loadFromFile(dataDir,inFile=os.path.join(os.path.dirname(args.reload),"class.csv"))
        else:
            bGen.loadAndSaveDir(dataDir,outFile=os.path.join(args.saveFolder,"class.csv"),trainFrac=0.9)
        net = FaceNet(isTraining=True,args=args)
        net.train(bGen)
