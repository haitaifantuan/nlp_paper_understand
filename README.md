# NLP_Paper_Understand

&ensp;&ensp;&ensp;&ensp;在这个repository里, 我会复现论文的代码。这些代码有可能是我自己写的，也有可能是论文作者分享的。如果代码是来自于论文作者的话，我有可能会重组一下代码，添加一些中文注释，从而让读者更容易读懂代码。

## 第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate


&ensp;&ensp;&ensp;&ensp;这篇就是有名的加入了attention机制的机器翻译模型。

&ensp;&ensp;&ensp;&ensp;该代码全部由我自己从0开始写的，如果写的有异味，望海涵。然后给我留言提一些建议。

&ensp;&ensp;&ensp;&ensp;代码使用的数据集是TED的翻译数据，在这里我提供了腾讯云盘的下载链接。

&ensp;&ensp;&ensp;&ensp;链接：https://share.weiyun.com/5TTvAFU

&ensp;&ensp;&ensp;&ensp;密码：f8jl9e

#### 论文原理说明以及代码详细讲解

&ensp;&ensp;&ensp;&ensp;代码中有具体的注释

&ensp;&ensp;&ensp;&ensp;同时，在我的博客上，可以看到代码详细的讲解以及论文的原理，大家戳这里：

&ensp;&ensp;&ensp;&ensp;https://blog.csdn.net/u011559882/article/details/102337490

#### 代码说明

&ensp;&ensp;&ensp;&ensp;**运行环境：**

&ensp;&ensp;&ensp;&ensp;python 3.6.7

&ensp;&ensp;&ensp;&ensp;tensorflow 1.14

&ensp;&ensp;&ensp;&ensp;在环境方面如果碰到了问题，大家可以使用Anaconda创建一个与我一模一样的环境。Anaconda管理环境还是非常方便的。


&ensp;&ensp;&ensp;&ensp;**运行步骤：**

&ensp;&ensp;&ensp;&ensp;有两种方法运行这个模型，一种是从0开始训练，一种是从我训练了50epoch上的模型开始训练。

&ensp;&ensp;&ensp;&ensp;从0开始训练的请看这里。

&ensp;&ensp;&ensp;&ensp;1.将整个repository克隆下来。

&ensp;&ensp;&ensp;&ensp;2.从腾讯云下载数据集，将名为“data.zip”的压缩文件解压到“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”文件夹下面。此时会多出一个名为“data”的文件夹，里面放有原始语料

&ensp;&ensp;&ensp;&ensp;3.如果使用vs code的话，那就使用vs code打开“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”文件夹。这里说明下，为什么打开的是“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”这个文件夹。因为我使用的绝对路径都是以“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”这个文件夹为当前路径的。

&ensp;&ensp;&ensp;&ensp;如果有人使用cmd运行，那也可以cd到这个文件夹。

&ensp;&ensp;&ensp;&ensp;4.然后运行名为train.py模块，便可以自动的开始训练训练集然后保存模型。

&ensp;&ensp;&ensp;&ensp;5.训练完成后，运行inference.py模块，便可以开启预测模块。

&ensp;&ensp;&ensp;&ensp;如果想要从我训练的模型上继续训练，或者直接进行inference操作，那就看这里。

&ensp;&ensp;&ensp;&ensp;1.将整个repository克隆下来。

&ensp;&ensp;&ensp;&ensp;2.从腾讯云下载数据集，将名为“已训练50个epoch的模型”的压缩文件解压到“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”文件夹下面。此时会多出一个名为“data”的文件夹，和"saved_things"文件夹。

&ensp;&ensp;&ensp;&ensp;3.如果使用vs code的话，那就使用vs code打开“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”文件夹。这里说明下，为什么打开的是“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”这个文件夹。因为我使用的绝对路径都是以“第4篇-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate”这个文件夹为当前路径的。

&ensp;&ensp;&ensp;&ensp;4.然后运行名为train.py模块，便可以自动加载我之前训练到好的模型继续训练。或者直接运行inference.py模块，便可以开启预测模块。

&ensp;&ensp;&ensp;&ensp;5.如果选择在我训练好的模型上继续训练，那么训练完成后，直接运行inference.py模块，便可以开启预测模块。

&ensp;&ensp;&ensp;&ensp;其他注意事项：

&ensp;&ensp;&ensp;&ensp;1.里面的超参是可以调整的，具体可以看train_args.py模块里面的注释。

&ensp;&ensp;&ensp;&ensp;2.在训练模型训练到一半的时候，如果需要停止训练，直接停止就可以了。下次直接打开继续训练就可以了，从上一个epoch开始继续训练。






## 第3篇-doc2vec-Distributed_Representations_of_Sentences_and_Documents

&ensp;&ensp;&ensp;&ensp;该代码全部由我自己从0开始写的，如果写的有异味，望海涵。然后给我留言提一些建议。

&ensp;&ensp;&ensp;&ensp;代码使用的数据集是aclImdb，在这里我提供了腾讯云盘的下载链接。

&ensp;&ensp;&ensp;&ensp;链接：https://share.weiyun.com/58cHCoc 

&ensp;&ensp;&ensp;&ensp;密码：g4jpl7

#### 论文原理说明

&ensp;&ensp;&ensp;&ensp;在我的博客上，可以看到论文的原理，大家戳这里：https://blog.csdn.net/u011559882/article/details/101231855

#### 代码说明

&ensp;&ensp;&ensp;&ensp;**运行环境：**

&ensp;&ensp;&ensp;&ensp;python 3.6.7

&ensp;&ensp;&ensp;&ensp;tensorflow 1.14

&ensp;&ensp;&ensp;&ensp;在环境方面如果碰到了问题，大家可以使用Anaconda创建一个与我一模一样的环境。Anaconda管理环境还是非常方便的。


&ensp;&ensp;&ensp;&ensp;**运行步骤：**

&ensp;&ensp;&ensp;&ensp;1.将整个repository克隆下来。

&ensp;&ensp;&ensp;&ensp;2.从腾讯云下载数据集，解压到“第3篇-doc2vec”的根目录下面。

&ensp;&ensp;&ensp;&ensp;3.运行main.py模块，便可以自动的开始训练训练集然后保存模型。

&ensp;&ensp;&ensp;&ensp;4.训练集训练完后，将main.py中的train_flag变量改为False，然后再次运行main.py，就可以对测试集进行测试了。

&ensp;&ensp;&ensp;&ensp;其他注意事项：

&ensp;&ensp;&ensp;&ensp;1.里面的超参是可以调整的，具体可以看main.py模块里面的注释。

&ensp;&ensp;&ensp;&ensp;2.在训练训练集的词向量和句向量的时候，训练到一半，如果需要停止训练，直接停止就可以了。下次直接打开继续训练就可以了，从上一个epoch开始继续训练。




## 第2篇-word2vec-Efficient_Estimation_of_Word_Representations_in_Vector_Space

&ensp;&ensp;&ensp;&ensp;本代码基于TensorFlow 的官方教程： https://www.tensorflow.org/tutorials/word2vec 。

&ensp;&ensp;&ensp;&ensp;我在官方教程的基础上进行了代码的修改，然后加了一些我个人的理解的注释，目的是为了让读者能够更容易的读懂代码。

&ensp;&ensp;&ensp;&ensp;代码使用的训练集为"text8"。代码里面已经写好下载它的代码，如果大家觉得在代码里下载的太慢，这里我提供了腾讯云盘的下载链接。

&ensp;&ensp;&ensp;&ensp;链接：https://share.weiyun.com/539UrqX 

&ensp;&ensp;&ensp;&ensp;密码：yajzmr

#### 论文原理说明

&ensp;&ensp;&ensp;&ensp;在我的博客上，可以看到论文的原理，大家戳这里：https://blog.csdn.net/u011559882/article/details/100678714

#### 代码说明

&ensp;&ensp;&ensp;&ensp;代码部分一共有两个文件，一个是“word2vec_CBOW.py”，一个是“word2vec_skip-gram.py”，大家从名字上也可以看出，一个是CBOW模型的代码，一个是skip-gram的代码。

&ensp;&ensp;&ensp;&ensp;**运行环境：**

&ensp;&ensp;&ensp;&ensp;python 3.6.7

&ensp;&ensp;&ensp;&ensp;tensorflow 1.14

&ensp;&ensp;&ensp;&ensp;在环境方面如果碰到了问题，大家可以使用Anaconda创建一个与我一模一样的环境。Anaconda管理环境还是非常方便的。

&ensp;&ensp;&ensp;&ensp;**运行步骤：**

&ensp;&ensp;&ensp;&ensp;1.从腾讯云盘下载“text8.zip”将其放到和代码同一个路径下的文件夹，也就是“第2篇-word2vec”这个文件夹下面。（或者这一步不执行，代码里也可以自动下载）

&ensp;&ensp;&ensp;&ensp;2.直接运行代码就可以了

&ensp;&ensp;&ensp;&ensp;3.运行完后，会在代码文件夹下面生成“wordvec_visualization.png”文件，这个是可视化文件，可以随意删除，没关系的。


