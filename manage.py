"""
Usage:
    manage.py (train) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>]
    manage.py (test) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--num=<num>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. 
    --js             Use physical joystick.
"""
'''
python package pandas, pillow, docopt, opencv-python are needed.
'''
import os
from docopt import docopt

from LRSDK.dataBuild import TubGroup
from KerasBaseModel1 import DriveModel

from keras.preprocessing.image import img_to_array, load_img

import numpy as np
import random
import json

def linear_bin(a):
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr

def linear_unbin(arr):
    # Convert a categorical array to value.
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def train(tub_names, new_model_path, base_model_path=None):
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def train_record_transform(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        record['user/throttle'] = round(record['user/throttle'],4)
        return record

    def val_record_transform(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        record['user/throttle'] = round(record['user/throttle'],4)
        return record

    new_model_path = os.path.expanduser(new_model_path)

    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        drive_model = DriveModel(base_model_path)
    else:
        drive_model = DriveModel()

    print('tub_names', tub_names)
    tubgroup = TubGroup(tub_names)

    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    train_record_transform=train_record_transform,
                                                    val_record_transform=val_record_transform,
                                                    batch_size=128,
                                                    train_frac=0.8)

    total_records = len(tubgroup.df)
    total_train = int(total_records * 0.8)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // 128
    print('steps_per_epoch', steps_per_epoch)

    drive_model.train(train_gen,
             val_gen,
             saved_model_path=new_model_path,
             steps=steps_per_epoch,
             train_split=0.8)

def test(tub_names, model_path, test_num = None):

    model_path = os.path.expanduser(model_path)


    drive_model = DriveModel(model_path=model_path)

    picList = os.listdir(tub_names)
    total = int((len(picList)-1)/2)
    
    if test_num == None:
        r = range(0,total)
    else:
        r = range(0,int(test_num))

        #r = random.sample(range(0,total), int(test_num))
    
    count = 0
    num = 0
    print('------开始测试------')
    for i in r:
        img = load_img(tub_names + '/'+ str(i) + '_cam-image_array_.jpg',target_size=(224,224,3))
        img = img_to_array(img)
        angle, throttle = drive_model.predict_image(img)
        
        f = open(tub_names + '/record_'+ str(i) + '.json','r')
        setting = json.load(f)
        y = setting['user/angle']
        y = linear_bin(y)
        y = linear_unbin(y)
        
        num +=1 
        if abs(round(angle,4)-round(y,4))<=0.29:
            count += 1
        print(str(i) + '_cam-image_array_.jpg    '+'预测值为: %.3f'%angle + ',真实值为: %.3f'%y + ' 准确率为:%.2f%%'%(count/num*100))

    acc = count/len(r)
    print('------测试结束------')
    print('平均准确率为:%.2f%%'%(acc*100))
        


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['train']:
        tub = args['--tub']
        new_model_path = args['--model']
        base_model_path = args['--base_model']
        train(tub, new_model_path, base_model_path)
        
    elif args['test']:
        tub = args['--tub']
        model_path = args['--model']
        test_num = args['--num']
        test(tub, model_path, test_num)





