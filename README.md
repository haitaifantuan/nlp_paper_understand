# NLP_Paper_Understand
In this repository, I'll replicate the code for every paper. The codes might be written by myself, or copy from the author. If the codes are from author, I'll recombine the codes and annotate the codes more explicitly so that readers are able to read the codes more easily.
## 第2篇-word2vec
&ensp;&ensp;&ensp;&ensp;代码使用的训练集为"text8"。代码里面已经写好下载它的代码，如果大家觉得在代码里下载的太慢，这里我提供了腾讯云盘的下载链接。

&ensp;&ensp;&ensp;&ensp;链接：https://share.weiyun.com/539UrqX 

&ensp;&ensp;&ensp;&ensp;密码：yajzmr
#### 论文原理说明
&ensp;&ensp;&ensp;&ensp;在我的博客上，可以看到论文的原理，大家戳这里：https://blog.csdn.net/u011559882/article/details/100678714

#### 代码说明
&ensp;&ensp;&ensp;&ensp;代码部分一共有两个文件，一个是“word2vec_CBOW.py”，一个是“word2vec_skip-gram.py”，大家从名字上也可以看出，一个是CBOW模型的代码，一个是skip-gram的代码。

&ensp;&ensp;&ensp;&ensp;运行步骤：

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;1.从腾讯云盘下载text8.zip将其放到和代码同一个路径下的文件夹。（或者这一步不执行，代码里也可以自动下载）

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;2.直接运行代码就可以了

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;3.运行完后，会在代码文件夹下面生成“wordvec_visualization.png”文件，这个是可视化文件，可以随意删除，没关系的。
#### 其他说明
本代码是来自于TensorFlow 的官方教程： https://www.tensorflow.org/tutorials/word2vec 。

我在官方教程的基础上进行了代码的修改，然后加了一些我个人的理解的注释，目的是为了让读者能够更容易的读懂代码。


## 第3篇-doc2vec
&ensp;&ensp;&ensp;&ensp;代码使用的数据集是aclImdb，在这里我提供了腾讯云盘的下载链接。
