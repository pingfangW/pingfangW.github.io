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
