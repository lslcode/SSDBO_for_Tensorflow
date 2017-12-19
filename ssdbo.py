"""
date: 2017/11/10
author: lslcode [jasonli8848@qq.com]
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

class SSDBO:
    def __init__(self, tf_sess, isTraining):
        # tensorflow session
        self.sess = tf_sess
        # 是否训练
        self.isTraining = isTraining
        # 允许的图像大小
        self.img_size = [300, 300]
        # 分类总数量
        self.classes_size = 21
        # 背景分类的值
        self.background_classes_val = 0
        # 每个特征图单元的default box数量
        self.default_box_size = [5, 5, 5, 5, 5, 5]
        # default box 尺寸长宽比例
        self.box_aspect_ratio = [1.0, 1.0/1.5, 1.5/1.0, 1.0/2.0, 2.0/1.0]
        # 卷积步长
        self.conv_strides = [1, 1, 1, 1]
        # 池化窗口
        self.pool_size = [1, 2, 2, 1]
        # 池化步长
        self.pool_strides = [1, 2, 2, 1]
        # Jaccard相似度判断阀值
        self.jaccard_value = 0.5

        # 初始化Tensorflow Graph
        self.generate_graph()

    def generate_graph(self):
        # 输入数据
        self.input = tf.placeholder(shape=[None, self.img_size[0], self.img_size[1], 3], dtype=tf.float32, name='input_image')
        # vvg16卷积层 1
        self.conv_1_1 = self.convolution(self.input, [3, 3,  3, 64], self.conv_strides,'conv_1_1')
        self.conv_1_2 = self.convolution(self.conv_1_1, [3, 3, 64, 64], self.conv_strides,'conv_1_2')
        self.conv_1_2 = tf.nn.max_pool(self.conv_1_2, self.pool_size, self.pool_strides, padding='SAME', name='pool_1_2')
        print('##   conv_1_2 shape: ' + str(self.conv_1_2.get_shape().as_list()))
        # vvg16卷积层 2
        self.conv_2_1 = self.convolution(self.conv_1_2, [3, 3,  64, 128], self.conv_strides,'conv_2_1')
        self.conv_2_2 = self.convolution(self.conv_2_1, [3, 3, 128, 128], self.conv_strides,'conv_2_2')
        self.conv_2_2 = tf.nn.max_pool(self.conv_2_2, self.pool_size, self.pool_strides, padding='SAME',   name='pool_2_2')
        print('##   conv_2_2 shape: ' + str(self.conv_2_2.get_shape().as_list()))
        # vvg16卷积层 3
        self.conv_3_1 = self.convolution(self.conv_2_2, [3, 3, 128, 256], self.conv_strides,'conv_3_1')
        self.conv_3_2 = self.convolution(self.conv_3_1, [3, 3, 256, 256], self.conv_strides,'conv_3_2')
        self.conv_3_3 = self.convolution(self.conv_3_2, [3, 3, 256, 256], self.conv_strides,'conv_3_3')
        self.conv_3_3 = tf.nn.max_pool(self.conv_3_3, self.pool_size, self.pool_strides, padding='SAME', name='pool_3_3')
        print('##   conv_3_3 shape: ' + str(self.conv_3_3.get_shape().as_list()))
        # vvg16卷积层 4
        self.conv_4_1 = self.convolution(self.conv_3_3, [3, 3, 256, 512], self.conv_strides,'conv_4_1')
        self.conv_4_2 = self.convolution(self.conv_4_1, [3, 3, 512, 512], self.conv_strides,'conv_4_2')
        self.conv_4_3 = self.convolution(self.conv_4_2, [3, 3, 512, 512], self.conv_strides,'conv_4_3')
        self.conv_4_3 = tf.nn.max_pool(self.conv_4_3, self.pool_size, self.pool_strides, padding='SAME', name='pool_4_3')
        print('##   conv_4_3 shape: ' + str(self.conv_4_3.get_shape().as_list()))
        # vvg16卷积层 5
        self.conv_5_1 = self.convolution(self.conv_4_3, [3, 3, 512, 512], self.conv_strides,'conv_5_1')
        self.conv_5_2 = self.convolution(self.conv_5_1, [3, 3, 512, 512], self.conv_strides,'conv_5_2')
        self.conv_5_3 = self.convolution(self.conv_5_2, [3, 3, 512, 512], self.conv_strides,'conv_5_3')
        self.conv_5_3 = tf.nn.max_pool(self.conv_5_3, self.pool_size, self.pool_strides, padding='SAME', name='pool_5_3')
        print('##   conv_5_3 shape: ' + str(self.conv_5_3.get_shape().as_list()))
        # 卷积层 6
        self.conv_6_1 = self.convolution(self.conv_5_3, [3, 3, 512, 1024], self.conv_strides,'conv_6_1')
        self.conv_6_1 = tf.nn.max_pool(self.conv_6_1, self.pool_size, self.pool_strides, padding='SAME', name='pool_6_1')
        print('##   conv_6_1 shape: ' + str(self.conv_6_1.get_shape().as_list()))
        # 卷积层 7 
        self.conv_7_1 = self.convolution(self.conv_6_1, [2, 2, 1024, 1024], self.conv_strides,'conv_7_1')
        self.conv_7_1 = tf.nn.max_pool(self.conv_7_1, self.pool_size, self.pool_strides, padding='SAME', name='pool_7_1')
        print('##   conv_7_1 shape: ' + str(self.conv_7_1.get_shape().as_list()))
        # 卷积层 8
        self.conv_8_1 = self.convolution(self.conv_7_1, [1, 1, 1024, 512], self.conv_strides,'conv_8_1')
        self.conv_8_1 = tf.nn.max_pool(self.conv_8_1, self.pool_size, self.pool_strides, padding='SAME',  name='pool_8_1')
        print('##   conv_8_1 shape: ' + str(self.conv_8_1.get_shape().as_list()))
        # 卷积层 9
        self.conv_9_1 = self.convolution(self.conv_8_1, [1, 1, 512, 512], self.conv_strides,'conv_9_1')
        self.conv_9_1 = tf.nn.max_pool(self.conv_9_1, self.pool_size, self.pool_strides, padding='SAME',  name='pool_9_1')
        print('##   conv_9_1 shape: ' + str(self.conv_9_1.get_shape().as_list()))

        # 第 1 层 特征层，来源于conv_4_3  
        self.features_1 = self.convolution(self.conv_4_3, [3, 3, 512, self.default_box_size[0] * self.classes_size], self.conv_strides,'features_1')
        print('##   features_1 shape: ' + str(self.features_1.get_shape().as_list()))
        # 第 2 层 特征层，来源于conv_5_3
        self.features_2 = self.convolution(self.conv_5_3, [3, 3, 512, self.default_box_size[1] * self.classes_size], self.conv_strides,'features_2')
        print('##   features_2 shape: ' + str(self.features_2.get_shape().as_list()))
        # 第 3 层 特征层，来源于conv_6_1
        self.features_3 = self.convolution(self.conv_6_1, [3, 3, 1024, self.default_box_size[2] * self.classes_size], self.conv_strides,'features_3')
        print('##   features_3 shape: ' + str(self.features_3.get_shape().as_list()))
        # 第 4 层 特征层，来源于conv_7_1
        self.features_4 = self.convolution(self.conv_7_1, [3, 3, 1024, self.default_box_size[3] * self.classes_size], self.conv_strides,'features_4')
        print('##   features_4 shape: ' + str(self.features_4.get_shape().as_list()))
        # 第 5 层 特征层，来源于conv_8_1
        self.features_5 = self.convolution(self.conv_8_1, [3, 3, 512, self.default_box_size[4] * self.classes_size], self.conv_strides,'features_5')
        print('##   features_5 shape: ' + str(self.features_5.get_shape().as_list()))
        # 第 6 层 特征层，来源于conv_9_1
        self.features_6 = self.convolution(self.conv_9_1, [3, 3, 512, self.default_box_size[4] * self.classes_size], self.conv_strides,'features_6')
        print('##   features_6 shape: ' + str(self.features_6.get_shape().as_list()))
        
        # 特征层集合
        self.feature_maps = [self.features_1, self.features_2, self.features_3, self.features_4, self.features_5, self.features_6]
        # 获取卷积后各个特征层的shape,以便生成feature和groundtruth格式一致的训练数据
        self.feature_maps_shape = [m.get_shape().as_list() for m in self.feature_maps]
        
        # 整理feature数据
        self.feature_class = []
        for i, fmap in zip(range(len(self.feature_maps)), self.feature_maps):
            width = self.feature_maps_shape[i][1]
            height = self.feature_maps_shape[i][2]
            self.feature_class.append(tf.reshape(fmap, [-1, (width * height * self.default_box_size[i]) , self.classes_size]))
        # 合并每张图像产生的所有特征
        self.feature_class = tf.concat(self.feature_class, axis=1)
        print('##   feature_class shape : ' + str(self.feature_class.get_shape().as_list()))

        # 生成所有default boxs
        self.all_default_boxs = self.generate_all_default_boxs()
        self.all_default_boxs_len = len(self.all_default_boxs)
        print('##   all default boxs : ' + str(self.all_default_boxs_len))

        # 输入真实数据
        self.groundtruth_class = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.int32,name='groundtruth_class')
        self.groundtruth_positives = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.float32,name='groundtruth_positives')
        self.groundtruth_positives_weight = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.float32,name='groundtruth_positives_weight')
        self.groundtruth_negatives = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.float32,name='groundtruth_negatives')

        # 损失函数
        self.groundtruth_count = tf.add(self.groundtruth_positives, self.groundtruth_negatives)
        self.groundtruth_count_weight = tf.add(self.groundtruth_positives_weight, self.groundtruth_negatives)
        self.softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.feature_class, labels=self.groundtruth_class)
        self.loss_all = tf.div(tf.reduce_sum(tf.multiply(self.softmax_cross_entropy, self.groundtruth_count_weight), reduction_indices=1) , tf.reduce_sum(self.groundtruth_count, reduction_indices = 1))
        
        # loss优化函数
        self.optimizer = tf.train.AdamOptimizer(0.001)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss_all)

    # 图像检测与训练
    # input_images : 输入图像数据，格式:[None,width,hight,channel]
    # actual_data : 标注数据，格式:[None,[None,center_X,center_Y,width,hight,classes]] , classes值范围[0,classes_size)
    def run(self, input_images, actual_data):
        # 训练部分
        if self.isTraining :
            if actual_data is None :
                raise Exception('actual_data参数不存在!')
            if len(input_images) != len(actual_data):
                raise Exception('input_images 与 actual_data参数长度不对应!')

            f_class = self.sess.run(self.feature_class, feed_dict={self.input : input_images})
            #检查数据是否正确
            f_class = self.check_numerics(f_class,'预测集f_class')
               
            with tf.control_dependencies([self.feature_class]):
                gt_class,gt_positives,gt_positives_weight,gt_negatives = self.generate_groundtruth_data(actual_data)
                #print('gt_positives :【'+str(np.sum(gt_positives))+'|'+str(np.amax(gt_positives))+'|'+str(np.amin(gt_positives))+'】|gt_positives_weight :【'+str(np.sum(gt_positives_weight))+'|'+str(np.amax(gt_positives_weight))+'|'+str(np.amin(gt_positives_weight))+'】|gt_negatives : 【'+str(np.sum(gt_negatives))+'|'+str(np.amax(gt_negatives))+'|'+str(np.amin(gt_negatives))+'】')

                self.sess.run(self.train, feed_dict={
                    self.input : input_images,
                    self.groundtruth_class : gt_class,
                    self.groundtruth_positives : gt_positives,
                    self.groundtruth_positives_weight : gt_positives_weight,
                    self.groundtruth_negatives : gt_negatives
                })
                loss_all,softmax_cross, h_positives = self.sess.run([self.loss_all, self.softmax_cross_entropy, self.groundtruth_positives], feed_dict={
                    self.input : input_images,
                    self.groundtruth_class : gt_class,
                    self.groundtruth_positives : gt_positives,
                    self.groundtruth_positives_weight : gt_positives_weight,
                    self.groundtruth_negatives : gt_negatives
                })
                #检查数据是否正确
                loss_all = self.check_numerics(loss_all,'损失值loss_all') 

                return loss_all, f_class ,softmax_cross, h_positives
        # 检测部分
        else :
            # softmax归一化预测结果
            feature_class_softmax = tf.nn.softmax(logits=self.feature_class, dim=-1)
            # 过滤background的预测值
            background_filter = np.ones(self.classes_size, dtype=np.float32)
            background_filter[self.background_classes_val] = 0 
            background_filter = tf.constant(background_filter)  
            feature_class_softmax=tf.multiply(feature_class_softmax, background_filter)
            # 计算每个box的最大预测值
            feature_class_softmax = tf.reduce_max(feature_class_softmax, 2)
            # 过滤冗余的预测结果
            box_top_set = tf.nn.top_k(feature_class_softmax, int(self.all_default_boxs_len/10))
            box_top_index = box_top_set.indices
            box_top_value = box_top_set.values
            f_class, f_class_softmax, box_top_index, box_top_value = self.sess.run(
                [self.feature_class, feature_class_softmax, box_top_index, box_top_value], 
                feed_dict={self.input : input_images}
            )
            top_shape = np.shape(box_top_index)
            pred_class = []
            pred_class_val = []
            pred_location = []
            for i in range(top_shape[0]) :
                item_img_class = []
                item_img_class_val = []
                item_img_location = []
                for j in range(top_shape[1]) : 
                    p_class_val = f_class_softmax[i][box_top_index[i][j]]
                    if p_class_val < 0.5:
                        continue
                    p_class = np.argmax(f_class[i][box_top_index[i][j]])
                    if p_class==self.background_classes_val:
                        continue
                    p_location = self.all_default_boxs[box_top_index[i][j]]
                    is_box_filter = False
                    for f_index in range(len(item_img_class)) :
                        if self.jaccard(p_location,item_img_location[f_index]) > 0.3 and p_class == item_img_class[f_index] :
                            is_box_filter = True
                            break
                    if is_box_filter == False :
                        item_img_class.append(p_class)
                        item_img_class_val.append(p_class_val)
                        item_img_location.append(p_location)
                pred_class.append(item_img_class)
                pred_class_val.append(item_img_class_val)
                pred_location.append(item_img_location)
            return pred_class, pred_class_val, pred_location

    # 卷积操作
    def convolution(self, input, kernel, strides, name):
        with tf.variable_scope(name):
            weight = tf.get_variable(initializer=tf.random_normal(kernel, 0, 1), dtype=tf.float32, name=name+'_weight')
            bias = tf.get_variable(initializer=tf.random_normal(kernel[-1:], 0, 1), dtype=tf.float32, name=name+'_bias')
            result = tf.nn.conv2d(input, weight, strides, padding='SAME', name=name+'_conv')
            result = self.batch_normalization(result, name=name+'_batch_normal')
            result = tf.nn.relu(tf.add(result, bias), name=name+'_relu')
            return result

    # Batch Normalization算法
    def batch_normalization(self, input, name):
        with tf.variable_scope(name):
            bn_input_shape = input.get_shape() 
            moving_mean = tf.get_variable('bn_mean', bn_input_shape[-1:] , initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('bn_variance', bn_input_shape[-1:] , initializer=tf.ones_initializer, trainable=False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(input, list(range(len(bn_input_shape) - 1)), name='bn_moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, 0.99999),assign_moving_average(moving_variance, variance, 0.99999)]):
                    return tf.identity(mean), tf.identity(variance)
            #mean, variance = tf.cond(tf.cast(self.isTraining, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            mean, variance = tf.cond(tf.cast(True, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            beta = tf.get_variable('bn_beta', bn_input_shape[-1:] , initializer=tf.zeros_initializer)
            gamma = tf.get_variable('bn_gamma', bn_input_shape[-1:] , initializer=tf.ones_initializer)
            return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 0.00001)
    
    # 初始化、整理训练数据
    def generate_all_default_boxs(self):
        # 全部按照比例计算并生成一张图像产生的每个default box的坐标以及长宽
        # 用于后续的jaccard匹配
        all_default_boxes = []
        for map_shape in self.feature_maps_shape:
            width = int(map_shape[1])
            height = int(map_shape[2])
            # 每个cell的长、宽、面积
            cell_width = 1 / float(width)
            cell_height = 1 / float(height)
            cell_scale = (cell_width * cell_height)
            for x in range(width):
                for y in range(height):
                    for ratio in self.box_aspect_ratio:
                        center_x = (x / float(width)) + (cell_width / 2)
                        center_y = (y / float(height)) + (cell_height / 2)
                        box_width = np.sqrt(cell_scale * ratio)
                        box_height = np.sqrt(cell_scale / ratio)
                        all_default_boxes.append([center_x, center_y, box_width, box_height])
        all_default_boxes = np.array(all_default_boxes)
        #检查数据是否正确
        all_default_boxes = self.check_numerics(all_default_boxes,'all_default_boxes') 
        return all_default_boxes

    # 整理生成groundtruth数据
    def generate_groundtruth_data(self, input_actual_data):
        # 生成空数组，用于保存groundtruth
        input_actual_data_len = len(input_actual_data)
        gt_class = np.zeros((input_actual_data_len, self.all_default_boxs_len)) 
        gt_positives = np.zeros((input_actual_data_len, self.all_default_boxs_len))
        gt_positives_jacc = np.zeros((input_actual_data_len, self.all_default_boxs_len))
        gt_negatives = np.zeros((input_actual_data_len, self.all_default_boxs_len))
        background_jacc = max(0, (self.jaccard_value-0.3))
        # 初始化正例训练数据
        for img_index in range(input_actual_data_len):
            for pre_actual in input_actual_data[img_index]:
                gt_class_val = pre_actual[-1:][0]
                gt_box_val = pre_actual[:-1]
                for boxe_index in range(self.all_default_boxs_len):
                    jacc = self.jaccard(gt_box_val, self.all_default_boxs[boxe_index])
                    if jacc > self.jaccard_value or jacc == self.jaccard_value:
                        gt_class[img_index][boxe_index] = gt_class_val
                        gt_positives[img_index][boxe_index] = 1
                        gt_positives_jacc[img_index][boxe_index] = jacc
                        gt_negatives[img_index][boxe_index] = 0
            # 正负例比值 1:3
            gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
            if (np.isnan(gt_neg_end_count) or gt_neg_end_count == 0) :
                #print('【没有匹配jacc】:'+str(pre_actual))
                gt_neg_end_count = 1
            if (gt_neg_end_count+np.sum(gt_positives[img_index])) > self.all_default_boxs_len :
                gt_neg_end_count = self.all_default_boxs_len - np.sum(gt_positives[img_index])
            # 随机选择负例
            gt_neg_index = np.random.randint(low=0, high=self.all_default_boxs_len, size=gt_neg_end_count)
            for r_index in gt_neg_index:
                if gt_positives_jacc[img_index][r_index] < background_jacc : 
                    gt_class[img_index][r_index] = self.background_classes_val
                    gt_positives[img_index][r_index] = 0
                    gt_negatives[img_index][r_index] = 1
        # 计算损失权重
        gt_positives_weight = np.multiply((1 + gt_positives_jacc - self.jaccard_value) , gt_positives)
        # 检查数据是否正确
        gt_class = self.check_numerics(gt_class,'gt_class')
        gt_positives = self.check_numerics(gt_positives,'gt_positives')
        gt_positives_weight = self.check_numerics(gt_positives_weight,'gt_positives_weight')
        gt_negatives = self.check_numerics(gt_negatives,'gt_negatives') 

        return gt_class, gt_positives, gt_positives_weight, gt_negatives

    # jaccard算法
    # 计算IOU，rect1、rect2格式为[center_x,center_y,width,height]           
    def jaccard(self, rect1, rect2):
        x_overlap = max(0, (min(rect1[0]+(rect1[2]/2), rect2[0]+(rect2[2]/2)) - max(rect1[0]-(rect1[2]/2), rect2[0]-(rect2[2]/2))))
        y_overlap = max(0, (min(rect1[1]+(rect1[3]/2), rect2[1]+(rect2[3]/2)) - max(rect1[1]-(rect1[3]/2), rect2[1]-(rect2[3]/2))))
        intersection = x_overlap * y_overlap
        # 删除超出图像大小的部分
        rect1_width_sub = 0
        rect1_height_sub = 0
        rect2_width_sub = 0
        rect2_height_sub = 0
        if (rect1[0]-rect1[2]/2) < 0 : rect1_width_sub += 0-(rect1[0]-rect1[2]/2)
        if (rect1[0]+rect1[2]/2) > 1 : rect1_width_sub += (rect1[0]+rect1[2]/2)-1
        if (rect1[1]-rect1[3]/2) < 0 : rect1_height_sub += 0-(rect1[1]-rect1[3]/2)
        if (rect1[1]+rect1[3]/2) > 1 : rect1_height_sub += (rect1[1]+rect1[3]/2)-1
        if (rect2[0]-rect2[2]/2) < 0 : rect2_width_sub += 0-(rect2[0]-rect2[2]/2)
        if (rect2[0]+rect2[2]/2) > 1 : rect2_width_sub += (rect2[0]+rect2[2]/2)-1
        if (rect2[1]-rect2[3]/2) < 0 : rect2_height_sub += 0-(rect2[1]-rect2[3]/2)
        if (rect2[1]+rect2[3]/2) > 1 : rect2_height_sub += (rect2[1]+rect2[3]/2)-1
        area_box_a = (rect1[2]-rect1_width_sub) * (rect1[3]-rect1_height_sub)
        area_box_b = (rect2[2]-rect2_width_sub) * (rect2[3]-rect2_height_sub)
        union = area_box_a + area_box_b - intersection
        if intersection > 0 and union > 0 : 
            return intersection / union 
        else : 
            return 0

    # 检测数据是否正常
    def check_numerics(self, input_dataset, message):
        if str(input_dataset).find('Tensor') == 0 :
            input_dataset = tf.check_numerics(input_dataset, message)
        else :
            dataset = np.array(input_dataset)
            nan_count = np.count_nonzero(dataset != dataset) 
            inf_count = len(dataset[dataset == float("inf")])
            n_inf_count = len(dataset[dataset == float("-inf")])
            if nan_count>0 or inf_count>0 or n_inf_count>0:
                data_error = '【'+ message +'】出现数据错误！【nan：'+str(nan_count)+'|inf：'+str(inf_count)+'|-inf：'+str(n_inf_count)+'】'
                raise Exception(data_error) 
        return  input_dataset
