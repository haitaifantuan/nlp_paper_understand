# NLP_Paper_Understand
&ensp;&ensp;&ensp;&ensp;在这个repository里, 我会复现论文的代码。这些代码有可能是我自己写的，也有可能是论文作者分享的。如果代码是来自于论文作者的话，我有可能会重组一下代码，添加一些中文注释，从而让读者更容易读懂代码。

## 第2篇-word2vec

&ensp;&ensp;&ensp;&ensp;本代码是来自于TensorFlow 的官方教程： https://www.tensorflow.org/tutorials/word2vec 。

&ensp;&ensp;&ensp;&ensp;我在官方教程的基础上进行了代码的修改，然后加了一些我个人的理解的注释，目的是为了让读者能够更容易的读懂代码。

&ensp;&ensp;&ensp;&ensp;代码使用的训练集为"text8"。代码里面已经写好下载它的代码，如果大家觉得在代码里下载的太慢，这里我提供了腾讯云盘的下载链接。

&ensp;&ensp;&ensp;&ensp;链接：https://share.weiyun.com/539UrqX 

&ensp;&ensp;&ensp;&ensp;密码：yajzmr

#### 论文原理说明

&ensp;&ensp;&ensp;&ensp;在我的博客上，可以看到论文的原理，大家戳这里：https://blog.csdn.net/u011559882/article/details/100678714

#### 代码说明

&ensp;&ensp;&ensp;&ensp;代码部分一共有两个文件，一个是“word2vec_CBOW.py”，一个是“word2vec_skip-gram.py”，大家从名字上也可以看出，一个是CBOW模型的代码，一个是skip-gram的代码。

&ensp;&ensp;&ensp;&ensp;<font color=red>运行环境：</font>

&ensp;&ensp;&ensp;&ensp;python 3.6.7

&ensp;&ensp;&ensp;&ensp;tensorflow 1.14

&ensp;&ensp;&ensp;&ensp;在环境方面如果碰到了问题，大家可以使用Anaconda创建一个与我一模一样的环境。Anaconda管理环境还是非常方便的。

&ensp;&ensp;&ensp;&ensp;<font color=red>运行步骤：</font>

&ensp;&ensp;&ensp;&ensp;1.从腾讯云盘下载“text8.zip”将其放到和代码同一个路径下的文件夹，也就是“第2篇-word2vec”这个文件夹下面。（或者这一步不执行，代码里也可以自动下载）

&ensp;&ensp;&ensp;&ensp;2.直接运行代码就可以了

&ensp;&ensp;&ensp;&ensp;3.运行完后，会在代码文件夹下面生成“wordvec_visualization.png”文件，这个是可视化文件，可以随意删除，没关系的。



## 第3篇-doc2vec

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