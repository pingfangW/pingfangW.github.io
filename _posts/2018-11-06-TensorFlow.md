---
layout: post
title: "TensorFlow"
description: "TensorFlow"
categories: [机器学习]
tags: [机器学习]
comments: true
---
# 目录：

* Kramdown table of contents
{:toc .toc}

# 什么是TensorFlow

“TensorFlow是一个使用数据流图进行数值计算的开源软件库。图中的节点表示数学运算，而图边表示在它们之间传递的多维数据阵列（又称张量）。

灵活的体系结构允许你使用单个API将计算部署到桌面、服务器或移动设备中的一个或多个CPU或GPU。“

![]({{ site.url }}/assets/images/networks/TensorFlow_01.gif)


如果你之前曾经使用过numpy，那么了解TensorFlow将会是小菜一碟！numpy和TensorFlow之间的一个主要区别是TensorFlow遵循一个“懒惰”的编程范例。

它首先建立所有要完成的操作图形，然后当一个“会话”被调用时，它再“运行”图形。构建一个计算图可以被认为是TensorFlow的主要成分。

TensorFlow不仅仅是一个强大的神经网络库。它可以让你在其上构建其他机器学习算法，如决策树或k最近邻。

## 典型的“张量流”

每个库都有自己的“实施细节”，即按照其编码模式编写的一种方法。例如，在执行scikit-learn时，首先创建所需算法的对象，然后在训练集上构建一个模型，并对测试集进行预测。例如：

~~~ python 

# 定义算法的超参数

clf = svm.SVC(gamma=0.001m C=100.)

# 训练

clf.fit(X, y)

# 预测

clf.predict(X_test)
~~~

TensorFlow遵循一个“懒惰”的方法。

在TensorFlow中运行程序的通常工作流程如下所示：

1．建立一个计算图（http://t.cn/RYRNUS6）。这可以是TensorFlow支持的任何数学操作。

2．初始化变量。

3．创建会话。

4．在会话中运行图形。

5．关闭会话。


举个例子：

写一个小程序来添加两个数字

~~~~ python 

import tensorflow as tf

# 建计算图
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
addition = tf.add(a, b)

# 初始化变量
init = tf.initialize_all_variables()

# 创建对话，并允许计算图
with tf.Session() as sess:
	sess.run(init)
	print("Addition: %i" % sess.run(addition, feed_dict+{a:2, b:3}))
	
# 关闭对话
sess.close()
~~~



# TensorFlow的优点

使用TensorFlow的优点是：

1.它有一个直观的结构，因为顾名思义，它有一个“张量流”。 你可以很容易地看到图的每一个部分。

2.轻松地在CPU / GPU上进行分布式计算。

3.平台灵活性。你可以在任何地方运行模型，无论是在移动设备，服务器还是PC上。




参考文献：

https://mp.weixin.qq.com/s/SrGfpcgqxtPQUD-Y5rFhEw
