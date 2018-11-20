---
layout: post
title: "机器学习系列课程-TensorFlow"
description: "TensorFlow"
categories: [机器学习]
tags: [机器学习, TensorFlow]
redirect_from:
  - /2018/10/23/
---

# 目录

* Kramdown table of contents
{:toc .toc}

# 正文

![]({{ site.url }}/assets/images/machinelearning/tensor_01.jpg)

![]({{ site.url }}/assets/images/machinelearning/tensor_02.jpg)

~~~
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
x = tf.get_variable("x", shape=(), dtype=tf.float32)
f = x**2

optimizer = tf.train.GradientDesecentOptimizer(0.1)
step = optimizer.minimize(f, var_list=[x])

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(10):
	_, curr_x, curr_f = s.run([step, x, f])
	print(curr_x, curr_f)

tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)
summaries = tf.summary.merge_all()

s = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter("logs/1", s.graph)
s.run(tf.global_variables_initializer())
for i in range(10):
	_,curr_summaries = s.run([step, summaries])
	summary_writer.add_summary(curr_summaries, i)
	summary_writer.flush()

~~~

TensorFlow实例：

1，计算0~N-1的平方和

> python版

~~~
import numpy as np

def sum_square(N):
	return np.sum(np.arange(N)**2)
	
sum_square(10)
~~~


> tensorflow版

~~~
import tensorflow as tf

tf.reset_default_graph()
s = tf.InteractiveSession()

N = tf.placeholder('int64', name="input_to_your_function")     
result = tf.reduce_sum(tf.range(N)**2)

s.run(result, {N:10})

writer = tf.summary.FileWriter("/tmp/tboard", graph=s.graph)
~~~

> 详细了解一下placeholder的用法

~~
with tf.name_scope("Placeholder_examples"):
	arbitrary_input = tf.placeholder('float32')
	input_vector = tf.placeholder('float32', shape=(None,))
	fixed_vector = tf.placeholder('int32', shape=(10,))
	input_matrix = tf.placeholder('float32', shape=(None, 15))
	
	input1 = tf.placeholder('float64', shape=(None, 100, None))
	input2 = tf.placeholder('int32', shape=(None, None, 3, 224, 224))
	
	double_the_vector = input_vector*2
	elementwise_cosine = tf.cos(input_vector)
	vector_squares = input_vector**2 - input_vector + 1
~~

> 再看一例，mean square error

~~
with tf.name_scope("MSE"):
	y_true = tf.placeholder("float32", shape=(None,), name="y_true")
	y__predicted = tf.placeholder("float32",shape=(None,), name="y_predicted")
	mse = tf.reduce_mean((y_true - y_predicted)**2)

def compute_mse(vector1, vector2):
	return mse.eval({y_true: vector1, y_predicted: vector2})
~~

> variable和placeholder不同的是，s.run()时，variable不需要传入数字。

~~
shared_vector_1 = tf.Variable(initial_value=np.ones(5), name="example_variable")   # 先创建一个变量

s.run(tf.global_variables_initializer())   # 初始化初始变量
print("initial value", s.run(shared_vector_1))

s.run(shared_vector_1.assign(np.arange(5)))
print("new value", s.run(shared_vector_1))
~~








