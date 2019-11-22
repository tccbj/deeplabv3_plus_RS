# -*- coding: utf-8 -*-

from osgeo import gdal
import os
import glob
import numpy as np
import tensorflow as tf
import collections
import six

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


def process(img_file, record_path, height, width):
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
    
    raw_height = img_raw.shape[0]
    raw_width = img_raw.shape[1]
    record_path = os.path.join(record_path, 'test-%s.tfrecord' % os.path.basename(img_file)[:-4])
    with tf.python_io.TFRecordWriter(record_path) as writer:
        count = 1
        for row in range(0, raw_height, height):
            for col in range(0, raw_width, width):
                img_tmp = img_raw[row:row+height, col:col+width, :]
                img_tmp = np.pad(img_tmp,((0,height-img_tmp.shape[0]),(0,width-img_tmp.shape[1]),(0,0)),'constant',constant_values=0)
                img_tmp = img_tmp.tobytes()
                tf_example = tf.train.Example(features = tf.train.Features(feature = {
                                            'image/encoded':_bytes_list_feature(img_tmp),
                                            # 'image/height': _int64_list_feature(height),
                                            # 'image/width': _int64_list_feature(width),
                                            #'image/segmentation/class/encoded':_bytes_list_feature(seg_tmp)
                                            }))

                writer.write(tf_example.SerializeToString())
                count += 1
    print('count',count)
    return count
    
if __name__=='__main__':
    record_path = r'G:\zj_yanhai\tfrecord'
    img_path = r'G:\zj_yanhai\export_output\Level16\*.tif'
    height = 321
    width = 321
    files = glob.glob(img_path)
    num = []
    for file in files:
        num.append(process(file, record_path, height, width))
    print('num: ',num)
    print('total: ',sum(num))