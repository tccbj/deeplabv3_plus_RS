# -*- coding: utf-8 -*-

from osgeo import gdal
import os
import glob
import numpy as np
import tensorflow as tf
import collections
import six
# 替换原来的build_data.py，适配自己的数据tif格式，tfrecord内容为image，width，height，seg

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_path',
                           'C:\\Users\\TL7050\\shuichan\\test\\image.tif',
                           'tif path')
                           
tf.app.flags.DEFINE_string('seg_path',
                           'C:\\Users\\TL7050\\shuichan\\test\\label.tif',
                           'seg label path')
                           
tf.app.flags.DEFINE_string('record_path',
                           'C:\\Users\\TL7050\\shuichan\\test',
                           'Path to save converted SSTable of TensorFlow examples.')
                           
tf.app.flags.DEFINE_integer('output_height',
                            321,
                            'patch height')
                            
tf.app.flags.DEFINE_integer('output_width',
                            321,
                            'patch width')
# 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)       #打开文件
    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    if len(im_data.shape)== 2:
        im_data = im_data[:,:,np.newaxis]
    else:
        im_data = im_data.transpose(1,2,0)  #转换成(width,height,channel)
    del dataset 
    return im_proj,im_geotrans,im_data
    

def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

    Args:
        values: A scalar or list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def process(img_file, seg_file, record_path, height, width):
    """
        converts one large tif image into tfrecord, each has a size of height*width
    
    Args:
        img_file: image filename.
        seg_file: seg filename.
        record_path: output record path.
        height: image height.
        width: image width

    """

    # img_file = img_file.decode()
    # seg_file = seg_file.decode()
    # record_path = record_path.decode()
    img_raw = read_img(img_file)[2]
    seg_raw = read_img(seg_file)[2]
    
    if img_raw.shape[:2] != seg_raw.shape[:2]:
        print(img_raw.shape,seg_raw.shape)
        return
    raw_height = img_raw.shape[0]
    raw_width = img_raw.shape[1]
    record_path_train = os.path.join(record_path, 'train-%s.tfrecord' % os.path.basename(img_file)[:-4])
    writer_train = tf.python_io.TFRecordWriter(record_path_train)
    record_path_val = os.path.join(record_path, 'val-%s.tfrecord' % os.path.basename(img_file)[:-4])
    writer_val = tf.python_io.TFRecordWriter(record_path_val)
#    with tf.python_io.TFRecordWriter(record_path) as writer:
    count_flag = 1
    count_train = 0
    count_val = 0
    
    for row in range(0, raw_height, height):
        for col in range(0, raw_width, width):
            img_tmp = img_raw[row:row+height, col:col+width, :]
            img_tmp = np.pad(img_tmp,((0,height-img_tmp.shape[0]),(0,width-img_tmp.shape[1]),(0,0)),'constant',constant_values=0)
            img_tmp_ud = np.flipud(img_tmp)
            img_tmp = img_tmp.tobytes()
            img_tmp_ud = img_tmp_ud.tobytes()
            seg_tmp = seg_raw[row:row+height, col:col+width, :]
            seg_tmp = np.pad(seg_tmp,((0,height-seg_tmp.shape[0]),(0,width-seg_tmp.shape[1]),(0,0)),'constant',constant_values=0)
            seg_tmp_ud = np.flipud(seg_tmp)
            seg_tmp = seg_tmp.tobytes()
            seg_tmp_ud = seg_tmp_ud.tobytes()
            tf_example = tf.train.Example(features = tf.train.Features(feature = {
                                        'image/encoded':_bytes_list_feature(img_tmp),
                                        # 'image/height': _int64_list_feature(height),
                                        # 'image/width': _int64_list_feature(width),
                                        'image/segmentation/class/encoded':_bytes_list_feature(seg_tmp)
                                        }))
            tf_example_ud = tf.train.Example(features = tf.train.Features(feature = {
                                        'image/encoded':_bytes_list_feature(img_tmp_ud),
                                        # 'image/height': _int64_list_feature(height),
                                        # 'image/width': _int64_list_feature(width),
                                        'image/segmentation/class/encoded':_bytes_list_feature(seg_tmp_ud)
                                        }))
            # if count_flag % 5 != 0:
            writer_train.write(tf_example.SerializeToString())
            writer_train.write(tf_example_ud.SerializeToString())
            count_train += 2
            # else:
                # writer_val.write(tf_example.SerializeToString())
                # writer_val.write(tf_example_ud.SerializeToString())
                # count_val += 2
            count_flag = (count_flag + 1) % 5
    writer_train.close()
    writer_val.close()
    print(img_file)
    print('train: ',count_train)
    print('val: ',count_val)
    return(count_train,count_val)
    
def process2(img_path, seg_path, record_path, height, width):
    """
        converts many large tif images into tfrecord, each has a size of height*width
    
    Args:
        img_file: image filename.
        seg_file: seg filename.
        record_path: output record path.
        height: image height.
        width: image width

    """
    
    record_path_train = os.path.join(record_path, 'train.tfrecord')
    writer_train = tf.python_io.TFRecordWriter(record_path_train)
    record_path_val = os.path.join(record_path, 'val.tfrecord')
    writer_val = tf.python_io.TFRecordWriter(record_path_val)
    img_files = glob.glob(img_path)
    seg_files = glob.glob(seg_path)
    files = zip(img_files,seg_files)
    count_flag = 1
    count_train = 0
    count_val = 0
    for file in files:
        img_file = file[0]
        seg_file = file[1]
        img_raw = read_img(img_file)[2]
        seg_raw = read_img(seg_file)[2]
        
        if img_raw.shape[:2] != seg_raw.shape[:2]:
            print(img_raw.shape,seg_raw.shape)
            return
        raw_height = img_raw.shape[0]
        raw_width = img_raw.shape[1]
        print(img_file,seg_file)
        for row in range(0, raw_height, height):
            for col in range(0, raw_width, width):
                img_tmp = img_raw[row:row+height, col:col+width, :]
                img_tmp = np.pad(img_tmp,((0,height-img_tmp.shape[0]),(0,width-img_tmp.shape[1]),(0,0)),'constant',constant_values=0)
                img_tmp_ud = np.flipud(img_tmp)
                img_tmp = img_tmp.tobytes()
                img_tmp_ud = img_tmp_ud.tobytes()
                seg_tmp = seg_raw[row:row+height, col:col+width, :]
                seg_tmp = np.pad(seg_tmp,((0,height-seg_tmp.shape[0]),(0,width-seg_tmp.shape[1]),(0,0)),'constant',constant_values=0)
                seg_tmp_ud = np.flipud(seg_tmp)
                seg_tmp = seg_tmp.tobytes()
                seg_tmp_ud = seg_tmp_ud.tobytes()
                tf_example = tf.train.Example(features = tf.train.Features(feature = {
                                            'image/encoded':_bytes_list_feature(img_tmp),
                                            # 'image/height': _int64_list_feature(height),
                                            # 'image/width': _int64_list_feature(width),
                                            'image/segmentation/class/encoded':_bytes_list_feature(seg_tmp)
                                            }))
                tf_example_ud = tf.train.Example(features = tf.train.Features(feature = {
                                            'image/encoded':_bytes_list_feature(img_tmp_ud),
                                            # 'image/height': _int64_list_feature(height),
                                            # 'image/width': _int64_list_feature(width),
                                            'image/segmentation/class/encoded':_bytes_list_feature(seg_tmp_ud)
                                            }))
                if count_flag % 5 != 0:
                    writer_train.write(tf_example.SerializeToString())
                    writer_train.write(tf_example_ud.SerializeToString())
                    count_train += 2
                else:
                    writer_val.write(tf_example.SerializeToString())
                    writer_val.write(tf_example_ud.SerializeToString())
                    count_val += 2
                # count_flag = (count_flag + 1) % 5
    writer_train.close()
    writer_val.close()
    
    print('train: ',count_train)
    print('val: ',count_val)
    return(count_train,count_val)
    
if __name__=='__main__':
    record_path = r'D:\1未完成项目\zj_sc\tfrecord'
    img_path = r'D:\1未完成项目\zj_sc\train_fine\tif\train_[0-9]*.tif'
    seg_path = r'D:\1未完成项目\zj_sc\train_fine\tif\train_gt*.tif'
    height = 321
    width = 321
    # img_files = glob.glob(img_path)
    # seg_files = glob.glob(seg_path)
    # files = zip(img_files,seg_files)
    # num = []
    # val = []
    # train = []
    # for file in files:
        # res = process(file[0], file[1], record_path, height, width)
        # train.append(res[0])
        # val.append(res[1])
    # print('num: ',num)
    # print('total val: ',sum(val))
    # print('total train: ',sum(train))
    train,val = process2(img_path, seg_path, record_path, height, width)
    # print(train,val)
    # img_file = r'D:\seg_fine\image.tif'
    # seg_file = r'D:\seg_fine\label.tif'
    # train,val = process(img_file, seg_file, record_path, height, width)