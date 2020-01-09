import os
import sys
import utils
import time
import numpy as np
import tensorflow as tf

class Resnet:
    def __init__(self,train_summary_step=50,val_summary_step=50,batch_size=200,max_steps=10000,saver_step=1000,weight_decay=0.0003):
        self.train_summary_step=train_summary_step
        self.val_summary_step=val_summary_step
        self.batch_size=batch_size
        self.max_steps=max_steps
        self.saver_step=saver_step
        self.weight_decay=weight_decay



    def conv_layer(self,input,output_channel,name,filter_size=3,stride=2):
        with tf.variable_scope(name) as scope:
          input_channel=input.shape.as_list()[-1]
          filter=tf.get_variable(initializer=tf.contrib.keras.initializers.he_normal(),shape=[filter_size,filter_size,input_channel,output_channel],name='filters')
          #filter=tf.Variable(tf.truncated_normal(stddev=1e-2,shape=[filter_size,filter_size,input_channel,output_channel]),name='filters')
          tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(self.weight_decay)(filter))
          conv=tf.nn.conv2d(input,filter,[1,stride,stride,1],padding='SAME')
          biases_constant=tf.constant(0.1,shape=[output_channel],dtype=tf.float32)
          biases=tf.Variable(biases_constant,name='biases')
          conv_biases=tf.add(conv,biases)
          batch_normal=self.batch_normal(conv_biases)
          relu=tf.nn.relu(batch_normal)
        return relu



    def build_graph(self):
        self.inputs=tf.placeholder(tf.float32,shape=[None,32,32,3])
        self.labels=tf.placeholder(tf.float32,shape=[None,10])

        self.global_step = tf.Variable(0 , trainable=False)
        self.train_flag=tf.placeholder(tf.bool)
        self.learning_rate=tf.placeholder(tf.float32)
        self.global_step=tf.Variable(0,trainable=False)



        conv1=self.conv_layer(self.inputs,64,'conv1',stride=1)

        cur_input=conv1
        output_channels=[128,256]
        j=0
        for i in range(2):
            for k in range(5):
              output=self.block(cur_input,output_channels[i],2,'block_{}'.format(j))
              cur_input=output
              j=j+1
        flatten=output
        flatten=tf.reduce_mean(flatten , [1 , 2])
        print(flatten.shape.as_list()[1])
        output=self.fc_layer(flatten,10,'output')

        loss_function=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.labels)
        self.loss_raw=tf.reduce_mean(loss_function)
        loss=self.loss_raw+self.l2_loss()
        optimizer=tf.train.MomentumOptimizer(self.learning_rate,0.9)
        grads=optimizer.compute_gradients(loss)
        self.train_op=optimizer.apply_gradients(grads,global_step=self.global_step)

        predictions=tf.nn.softmax(output)
        self.error=self.top1_error(predictions,self.labels)


        for gra,var in grads:
            name=var.op.name
            name=name.split('/')

            if name[-1]=='biases' or name[-1]=='filters' or name[-1]=='weights':
              tf.summary.histogram(var.op.name+'/gradients',gra)
        #tf.image_summary('images',self.inputs)
        for var in tf.trainable_variables():
            name=var.op.name
            #print(name)
            name=name.split('/')
            if name[-1]=='biases' or name[-1]=='filters' or name[-1]=='weights':
              tf.summary.histogram(var.op.name,var)

        tf.summary.scalar('loss',loss)
        tf.summary.scalar('error',self.error)
        self.summary_op=tf.summary.merge_all()




    def top1_error(self,predictions,labels):
        correct=tf.equal(tf.argmax(predictions,1),tf.argmax(labels,1))
        correct=tf.reduce_mean(tf.cast(correct,tf.float32))

        return 1-correct



    def block(self,input,output_channel,layer_size,name,filter_size=3,stride=1):

       input_channel = input.shape.as_list()[-1]
       if input_channel*2==output_channel:
         stride=2
       with tf.variable_scope(name) as scope:
          #for i in range(1,layer_size+1):
           output=self.conv_layer(input,output_channel,'conv_{}'.format(0),filter_size,stride)
           output=self.conv_layer(output,output_channel,'conv_{}'.format(1),filter_size,1)
           #print(output.shape.as_list())
       if input_channel*2 ==output_channel:
            input=tf.nn.avg_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            input=tf.pad(input,[[0,0],[0,0],[0,0],[np.int(input_channel/2),np.int(input_channel/2)]])
       output=output+input
       #output=self.max_pool(output,name)
       return output

    def max_pool(self , input , name,filter_size=3,stride=2):
        return tf.nn.max_pool(input , ksize=[1 , filter_size , filter_size , 1] , strides=[1 , stride , stride , 1] , padding='SAME' , name=name)

    def batch_normal(self , input):
        return tf.contrib.layers.batch_norm(input , decay=0.9 , center=True , scale=True , epsilon=1e-3 ,
                                            is_training=self.train_flag , updates_collections=None)

    def learing_rate_change(self,global_step):
        if global_step<10000:
            return 0.1
        elif global_step<15000:
            return 0.01
        else:
            return 0.001

    def l2_loss(self):
        weights = []
        #print(len(tf.trainable_variables()))
        for var in tf.trainable_variables():
            name = var.name
            name = name.split(':')[0]
            #print(name)
            name = name.split('/')[1]
            # if name=='biases' or name=='fc_biases':
            # continue
            weights.append(var)
        return self.weight_decay * tf.add_n([tf.nn.l2_loss(weight) for weight in weights])

    def train(self,resume=False):
        self.build_graph()
        cifbatch=utils.CifarBatch(self.batch_size)
        saver=tf.train.Saver()
        summary_write_train=tf.summary.FileWriter(os.path.join('summary1','train'))
        summary_write_val=tf.summary.FileWriter(os.path.join('summary1','val'))
        check_point_dir=os.path.join('checkpoint1')
        check_point_path=os.path.join(check_point_dir,'model.kpt')
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            if resume:
                latest = tf.train.latest_checkpoint(check_point_dir)
                if latest is None:
                    sys.exit(1)
                saver.restore(sess,latest)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            cifbatch.prepare_data()
            for s in range(self.max_steps):
                batch_data,batch_labels=cifbatch.get_next_batch_train(True)
                start=time.time()
                global_step=sess.run(self.global_step)
                feed_dict={self.inputs:batch_data,self.labels:batch_labels,self.train_flag:True,self.learning_rate:self.learing_rate_change(global_step)}
                _,loss,error,global_step=sess.run([self.train_op,self.loss_raw,self.error,self.global_step],feed_dict=feed_dict)
                end=time.time()


                if global_step%self.train_summary_step==0:
                     print("cost time:{}  loss:{}  error:{}  global_step:{}".format((end - start) , loss , error,global_step))
                     summary=sess.run(self.summary_op,feed_dict=feed_dict)
                     summary_write_train.add_summary(summary,global_step)
                     summary_write_train.flush()

                if global_step%self.val_summary_step==0:
                    val_batch_data,val_batch_labels=cifbatch.get_next_batch_val()
                    val_summary,val_loss,val_error=sess.run([self.summary_op,self.loss_raw,self.error],feed_dict={self.inputs:val_batch_data,self.labels:val_batch_labels,self.train_flag:False})
                    summary_write_val.add_summary(val_summary,global_step)
                    summary_write_val.flush()
                    print("val: loss:{} error:{}  global_step:{}".format(val_loss,val_error,global_step))

                if global_step%self.saver_step==0:
                    saver.save(sess,check_point_path,global_step)
    def test(self):
        self.build_graph()
        cifbatch=utils.CifarBatch(self.batch_size)
        saver=tf.train.Saver()
        check_point_dir=os.path.join('checkpoint1')
        check_point_path=os.path.join(check_point_dir,'model.kpt')
        test_loss=0
        test_error=0
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            latest = tf.train.latest_checkpoint(check_point_dir)
            if latest is None:
                print("读取模型失败")
                sys.exit(1)
            saver.restore(sess , latest)
            while True:
                test_batch_data,test_batch_labels=cifbatch.get_next_batch_test()
                if test_batch_data is None:
                   test_batch_num = cifbatch.get_num_batch_test()
                   print("test: loss{} error{}".format(test_loss/test_batch_num,test_error/test_batch_num))
                   break
                else:
                  loss , error = sess.run([self.loss_raw , self.error] ,
                                               feed_dict={self.inputs: test_batch_data , self.labels: test_batch_labels ,
                                                          self.train_flag: False})
                  test_loss+=loss
                  test_error+=error



    def fc_layer(self,input,output_size,name):
        with tf.variable_scope(name) as scope:
            #tf.contrib.keras.initializers.he_normal()
            weight=tf.get_variable(shape=[input.shape.as_list()[1],output_size],initializer=tf.contrib.keras.initializers.he_normal(),name='weights')
            #weight=tf.Variable(tf.truncated_normal(stddev=1e-2,shape=[input.shape.as_list()[1],output_size]),name='weights')
            tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(self.weight_decay)(weight))
            biases_constant = tf.constant(0.1 , shape=[output_size] , dtype=tf.float32)
            biases=tf.Variable(biases_constant,name='biases')
            output=tf.add(tf.matmul(input,weight),biases)
            output=self.batch_normal(output)
            output=tf.nn.relu(output)
        return output

if __name__=='__main__':
    res=Resnet(50,50,400,20000,1000)
    res.train()
    print("test")
    #res.test()
