# SSDBO_for_Tensorflow
<br/>
这套源码是 <a href='https://github.com/lslcode/SSD_for_Tensorflow' target='_blank'>SSD_for_Tensorflow</a> 的改良版本。
<br/><br/>
主要改进2个方面：<br/>
1，去除box offest的回归运算，改用default box的位置匹配目标；<br/>
2，class计算损失值时，正例计数增加jaccard值，作为激励权值，提高位置预测的准确度；<br/>
<br/><br/>

<b>为什么去除box offest？</b><br/>
标准版的ssd是对300 * 300的图片预设8732个default box，用于回归出预测box的偏移量。换句话说，这300 * 300的图片里面塞满了8732个框来匹配目标图像，这样的话总会有一个defualt box是几乎完全配对上的，因此我想是可以去除box offest的。虽然理论上会存在一点误差，但我想这点误差是可以接受的，因为即使算法可以做到100%准确预测位置，但是标注样本过程中也并不能保证100%没有出错。<br/>
另外，如果将box offest和class相加再一起回归，会相互干扰，容易出现梯度弥散现象。<br/><br/>

<b>为什么正例计数增加jaccard值？</b><br/>
在原版ssd中，box jaccard值大于0.5就认为是正例，也即是说 0.5<jaccard值<1 都被当作为正例来训练，因此最后class预测的结果只要是满足0.5<jaccard值<1 ，都认为是正确的class，这样会降低位置预测的正确率，例如标注半边脸的人脸预测。<br/>
因此正例计数增加jaccard值，即 positive = 1+jaccard-0.5 ，当jaccard=0.6时positive=1.1，当jaccard=0.9时positive=1.4，box匹配度越高positive激励值就越大。
