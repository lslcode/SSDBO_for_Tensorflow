"""
date: 2017/12/10
author: lslcode [jasonli8848@qq.com]
"""

import os
import gc
import xml.etree.ElementTree as etxml
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import ssdbo

'''
SSDBO检测
'''
def testing():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssdbo_model = ssdbo.SSDBO(sess,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            saver.restore(sess, './session_params/session.ckpt')
            image, actual,file_list = get_traindata_voc2007(1)
            pred_class, pred_class_val, pred_location = ssdbo_model.run(image,None)
            print('file_list:' + str(file_list))
             
            for index, act in zip(range(len(image)), actual):
                for a in act :
                    print('【img-'+str(index)+' actual】:' + str(a))
                '''    
                    for p_i, p_class in zip(range(len(pred_class[index])),pred_class[index]):
                        if(a[4]==p_class):
                            print('pred_class:' + str(p_class))
                            print('pred_class_val:' + str(pred_class_val[index][p_i]))
                            print('pred_location:' + str(pred_location[index][p_i]))
                '''
                print('pred_class:' + str(pred_class[index]))
                print('pred_class_val:' + str(pred_class_val[index]))
                print('pred_location:' + str(pred_location[index]))
                
                
              
        else:
            print('No Data Exists!')
        sess.close()
    
'''
SSDBO训练
'''
def training():
    batch_size = 15
    running_count = 0
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssdbo_model = ssdbo.SSDBO(sess,True)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            print('\nStart Restore')
            saver.restore(sess, './session_params/session.ckpt')
            print('\nEnd Restore')
         
        print('\nStart Training')
       
        min_loss_class = 10000000.
        while(min_loss_class > 0.0001 and running_count < 100000):
            running_count += 1
            
            train_data, actual_data,file_list = get_traindata_voc2007(batch_size)
            #print('file_list:' + str(file_list))
            #print('actual_data:' + str(actual_data))
            if len(train_data) > 0:
                loss_all, f_class, softmax_cross, h_positives = ssdbo_model.run(train_data, actual_data)
                loss_all = np.sum(loss_all)
                if min_loss_class > loss_all:
                    min_loss_class = loss_all

                #print('h_positives :【'+str(np.sum(h_positives))+'|'+str(np.amax(h_positives))+'|'+str(np.amin(h_positives))+'|'+str(np.mean(h_positives))+'】')
                #print('softmax_cross :【'+str(np.sum(softmax_cross))+'|'+str(np.amax(softmax_cross))+'|'+str(np.amin(softmax_cross))+'|'+str(np.mean(softmax_cross))+'】')
                
                count = 0;
                for h_i in range(len(train_data)):
                    for i_i in range(ssdbo_model.all_default_boxs_len):
                        if h_positives[h_i][i_i]==1:
                            print('softmax_cross_val :【'+str(softmax_cross[h_i][i_i])+'】【'+str(h_positives[h_i][i_i])+'】')
                            count+=1
                            if count>15:break
                    if count>15:break    
                count = 0;
                for h_i in range(len(train_data)):
                    for i_i in range(ssdbo_model.all_default_boxs_len):
                        if h_positives[h_i][i_i]==0:
                            print('softmax_cross_val :【'+str(softmax_cross[h_i][i_i])+'】')
                            count+=1
                            if count>15:break
                    if count>15:break 
               
                print('Running:【' + str(running_count) + '】|Loss All:【'+str(min_loss_class)+'|'+str(loss_all) + '】|pred_class:【'+ str(np.sum(f_class))+'|'+str(np.amax(f_class))+'|'+ str(np.amin(f_class))+'|'+ str(np.mean(f_class)) + '】')
                
                # 定期保存ckpt
                if running_count % 100 == 0:
                    saver.save(sess, './session_params/session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break
            
        saver.save(sess, './session_params/session.ckpt')
        sess.close()
        gc.collect()
            
    print('End Training')
    
'''
获取voc2007训练图片数据
train_data：训练批次图像，格式[None,width,height,3]
actual_data：图像标注数据，格式[None,[None,center_x,center_y,width,height,lable]]
'''
file_name_list = os.listdir('./train_datasets/voc2007/JPEGImages/')
lable_arr = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
# 图像白化，格式:[R,G,B]
#whitened_RGB_mean = [123.68, 116.78, 103.94]
def get_traindata_voc2007(batch_size):
    def get_actual_data_from_xml(xml_path):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_path).getroot()
            img_width =  float(annotation_node.find('size').find('width').text.strip())
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')       
            for obj_node in object_node_list:                       
                lable = lable_arr.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max)/2/img_width), ((y_min + y_max)/2/img_height), ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item  
        except:
            return None
        
    train_data = []
    actual_data = []
    
    file_list = random.sample(file_name_list, batch_size)
    
    for f_name in file_list :
        img_path = './train_datasets/voc2007/JPEGImages/' + f_name
        xml_path = './train_datasets/voc2007/Annotations/' + f_name.replace('.jpg','.xml')
        if os.path.splitext(img_path)[1].lower() == '.jpg' :
            actual_item = get_actual_data_from_xml(xml_path)
            if actual_item != None :
                actual_data.append(actual_item)
            else :
                print('Error : '+xml_path)
                continue
            img = skimage.io.imread(img_path)
            img = skimage.transform.resize(img, (300, 300))
            # 图像白化预处理
            #img = img - whitened_RGB_mean
            train_data.append(img)
            
    return train_data, actual_data,file_list

 
'''
主程序入口
'''
if __name__ == '__main__':
    print('\nStart Running')
    # 检测
    #testing()
    # 训练
    training()
    print('\nEnd Running')
   
        
