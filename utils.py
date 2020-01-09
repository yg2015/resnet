import os
import numpy as np
import random

class CifarBatch:
    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.train_cur_index=0
        self.val_cur_index=0
        self.test_cur_index=0
        self.test_load=False

    def data_preprocessing(self , x_data):

        x_data = x_data.astype('float32')

        x_data[: , : , : , 0] = (x_data[: , : , : , 0] - np.mean(x_data[: , : , : , 0])) / np.std(x_data[: , : , : , 0])
        x_data[: , : , : , 1] = (x_data[: , : , : , 1] - np.mean(x_data[: , : , : , 1])) / np.std(x_data[: , : , : , 1])
        x_data[: , : , : , 2] = (x_data[: , : , : , 2] - np.mean(x_data[: , : , : , 2])) / np.std(x_data[: , : , : , 2])

        return x_data

    def _random_crop(self , batch , crop_shape , padding=None):
        oshape = np.shape(batch[0])

        if padding:
            oshape = (oshape[0] + 2 * padding , oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding , padding) , (padding , padding) , (0 , 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i] , pad_width=npad ,
                                          mode='constant' , constant_values=0)
            nh = random.randint(0 , oshape[0] - crop_shape[0])
            nw = random.randint(0 , oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0] ,
                           nw:nw + crop_shape[1]]
        return new_batch

    def _random_flip_leftright(self , batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def data_augmentation(self , batch):
        batch = self._random_flip_leftright(batch)
        batch = self._random_crop(batch , [32 , 32] , 4)
        return batch

    def prepare_data(self):
        train_data = None
        train_labels = None
        test_data = None
        test_labels = None
        for i in range(1 , 6):
            batch_file = 'data_batch_' + str(i)
            batch_file_path = os.path.join('..' , 'cifar10-dataset' , batch_file)
            batch_dict = np.load(batch_file_path , encoding='bytes' , allow_pickle=True)
            data = batch_dict[b'data']
            labels = batch_dict[b'labels']
            if train_data is None:
                train_data = data
                train_labels = labels
            else:
                train_data = np.vstack((train_data , data))
                train_labels.extend(labels)
        self.train_labels = self.transfer_labels(train_labels)
        test_file_path = os.path.join('..' , 'cifar10-dataset' , 'test_batch')
        test_dict = np.load(test_file_path , encoding='bytes' , allow_pickle=True)
        test_data = test_dict[b'data']
        test_labels = test_dict[b'labels']
        self.test_labels = self.transfer_labels(test_labels)
        train_data = train_data.reshape(train_data.shape[0] , 3 , 32 , 32)
        train_data = train_data.transpose([0 , 2 , 3 , 1])
        self.train_data = self.data_preprocessing(train_data)
        test_data = test_data.reshape(test_data.shape[0] , 3 , 32 , 32)
        test_data = test_data.transpose([0 , 2 , 3 , 1])
        self.test_data = self.data_preprocessing(test_data)
        self.train_index_array=np.arange(len(self.train_data))
        self.test_index_array=np.arange(len(self.test_data))

    def get_next_batch_train(self,dataAugment=False):
        if self.train_cur_index>=len(self.train_data):
            self.train_cur_index=0
        if self.train_cur_index==0:
            np.random.shuffle(self.train_index_array)

        indexes=self.train_index_array[self.train_cur_index:self.train_cur_index+self.batch_size]
        self.train_cur_index=self.train_cur_index+self.batch_size
        batch_data=self.train_data[indexes]
        batch_labels=self.train_labels[indexes]
        if dataAugment:
            batch_data=self.data_augmentation(batch_data)
        return batch_data,batch_labels
    def get_next_batch_val(self):
        if self.val_cur_index>=len(self.test_data):
            self.val_cur_index=0
        if self.val_cur_index==0:
            np.random.shuffle(self.test_index_array)
        indexes=self.test_index_array[self.val_cur_index:self.val_cur_index+self.batch_size]
        self.val_cur_index=self.val_cur_index+self.batch_size
        batch_data=self.test_data[indexes]
        batch_labels=self.test_labels[indexes]
        return batch_data,batch_labels

    def transfer_labels(self , labels):
        labels_new = np.zeros((len(labels) , 10) , dtype='float32')
        for i in range(len(labels)):
            yaxis = labels[i]
            labels_new[i , yaxis] = 1
        return labels_new
    def get_num_batch_test(self):
        #assert  len(self.test_data)%self.batch_size==0 "测试数据集不能整除batch_size"
        return np.int(len(self.test_data)/self.batch_size)
    def get_next_batch_test(self):
        if self.test_load==False:
            test_file_path = os.path.join('..' , 'cifar10-dataset' , 'test_batch')
            test_dict = np.load(test_file_path , encoding='bytes' , allow_pickle=True)
            test_data=test_dict[b'data']
            test_data = test_data.reshape(test_data.shape[0] , 3 , 32 , 32)
            test_data = test_data.transpose([0 , 2 , 3 , 1])
            self.test_data = self.data_preprocessing(test_data)
            test_labels=test_dict[b'labels']
            self.test_labels = self.transfer_labels(test_labels)
            self.test_load=True
        if self.test_cur_index>=len(self.test_data):
            return None,None
        batch_data=self.test_data[self.test_cur_index:self.test_cur_index+self.batch_size]
        batch_labels=self.test_labels[self.test_cur_index:self.test_cur_index+self.batch_size]
        self.test_cur_index=self.test_cur_index+self.batch_size
        return batch_data,batch_labels
if __name__=='__main__':
    cf=CifarBatch(50)
    cf.prepare_data()
    for i in range(5):
      m,n=cf.get_next_batch_train()
      print("aa")
