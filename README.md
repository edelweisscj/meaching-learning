### Semi-supervised Learning on Graphs with Generative Adversarial Nets

### 1. Read the paper.

文章研究了生成对抗网（generative attersarial nets，GANs）如何帮助图的半监督学习。

文章翻译整理在https://blog.csdn.net/weixin_42281282/article/details/107575772

提出了图的半监督学习的一种新方法GraphSGAN。在GraphSGAN中，生成器和分类器网络进行了一种新颖的竞争博弈。在平衡状态下，生成器在子图之间的低密度区域生成假样本。为了区分真假样本，分类器隐式地考虑了子图的密度特性。提出了一种有效的对抗学习算法，在理论上保证了对传统规范化图拉普拉斯正则化的改进。在几种不同类型的数据集上的实验结果表明，所提出的GraphSGAN明显优于几种最新的方法。GraphSGAN还可以使用小批量进行训练，因此具有可伸缩性优势。

文章的贡献如下：

•引入了GANs作为解决半监督环境下图的分类问题的工具。GraphSGAN在图的低密度区域生成假样本，并利用聚类特性帮助分类。

•为GraphSGAN设计了一个新颖的发电机鉴别器之间的竞争博弈，深入分析了训练过程中的动力学、平衡和工作原理。此外，我们还对传统算法进行了改进。理论证明和实验验证都表明了该方法的有效性。

•在多个不同比例的数据集上评估我们的模型。GraphSGAN的性能明显优于以前的工作，并且展示了出色的可伸缩性。

### 2. Implement GraphSGAN on Cora dataset.
### 3. Compare your prediction results with the results reported in the paper.

（1）参考 https://github.com/THUDM/GraphSGAN 代码利用作业Cora数据集对文章进行了复现。参数选择基本参考论文。复现结果如下：

Iteration 0, loss_supervised = 0.0065, loss_unsupervised = 0.4171, loss_gen = 0.4028 train acc = 0.9986
Eval: correct 823 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 1, loss_supervised = 0.0055, loss_unsupervised = 0.4247, loss_gen = 0.4114 train acc = 0.9983
Eval: correct 819 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 2, loss_supervised = 0.0043, loss_unsupervised = 0.4391, loss_gen = 0.4180 train acc = 0.9991
Eval: correct 821 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 3, loss_supervised = 0.0060, loss_unsupervised = 0.4282, loss_gen = 0.4162 train acc = 0.9986
Eval: correct 822 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 4, loss_supervised = 0.0046, loss_unsupervised = 0.4295, loss_gen = 0.4117 train acc = 0.9983
Eval: correct 822 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 5, loss_supervised = 0.0054, loss_unsupervised = 0.4238, loss_gen = 0.4225 train acc = 0.9986
Eval: correct 816 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 6, loss_supervised = 0.0043, loss_unsupervised = 0.4345, loss_gen = 0.4318 train acc = 0.9995
Eval: correct 820 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 7, loss_supervised = 0.0044, loss_unsupervised = 0.4328, loss_gen = 0.4406 train acc = 0.9989
Eval: correct 816 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 8, loss_supervised = 0.0044, loss_unsupervised = 0.4293, loss_gen = 0.4385 train acc = 0.9986
Eval: correct 816 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 9, loss_supervised = 0.0043, loss_unsupervised = 0.4193, loss_gen = 0.4448 train acc = 0.9994
Eval: correct 816 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 10, loss_supervised = 0.0035, loss_unsupervised = 0.4171, loss_gen = 0.4534 train acc = 0.9994
Eval: correct 818 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 11, loss_supervised = 0.0053, loss_unsupervised = 0.4100, loss_gen = 0.4565 train acc = 0.9987
Eval: correct 817 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 12, loss_supervised = 0.0041, loss_unsupervised = 0.4040, loss_gen = 0.4559 train acc = 0.9992
Eval: correct 821 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 13, loss_supervised = 0.0062, loss_unsupervised = 0.3834, loss_gen = 0.4686 train acc = 0.9978
Eval: correct 820 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 14, loss_supervised = 0.0050, loss_unsupervised = 0.3781, loss_gen = 0.4723 train acc = 0.9989
Eval: correct 819 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 15, loss_supervised = 0.0046, loss_unsupervised = 0.3809, loss_gen = 0.4743 train acc = 0.9984
Eval: correct 821 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 16, loss_supervised = 0.0034, loss_unsupervised = 0.3799, loss_gen = 0.4726 train acc = 0.9992
Eval: correct 820 / 1000, Acc: 82.00
Training: 100 / 100
Iteration 17, loss_supervised = 0.0037, loss_unsupervised = 0.3762, loss_gen = 0.4831 train acc = 0.9991
Eval: correct 819 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 18, loss_supervised = 0.0035, loss_unsupervised = 0.3721, loss_gen = 0.4875 train acc = 0.9991
Eval: correct 819 / 1000, Acc: 81.00
Training: 100 / 100
Iteration 19, loss_supervised = 0.0027, loss_unsupervised = 0.3674, loss_gen = 0.4858 train acc = 0.9995
Eval: correct 814 / 1000, Acc: 81.00

以上可以看出，在Iteration 17之后，训练模型发生过拟合，即在训练集上准确率增大，但测试集评估指标下降。最终的classification accuracy (%)略低于paper中的83.0 ± 1.3



### 4. Analyze the effects of different losses.


在均衡状态下，没有一个player可以改变他们的策略来单方面减少他的损失。假设G在平衡时在中心区域生成样本，文中提出了D在最后表示层（n维高维空间）中达到预期平衡的四个条件：

（1） 不同类的节点应该映射到不同的集群中。（2） 标记和未标记的节点都不应映射到中心区域，以使其成为一个密度间隙。（3） 每个未标记的节点都应该映射到一个表示特定标签的集群中。（4） 不同的集群应该足够远。

 loss_supervised is defined as the cross entropy between predicted distribution over M classes and one-hot representation for real label.所以用来满足条件（1）。
 
 The classifier D incurs loss_unsupervised when real-or-fake misclassification happens.
条件（2）相当于原GAN中D的目的，因为g在中心密度隙中产生假样本。


loss_supervised = 0.0027, loss_unsupervised = 0.3674, loss_gen

### 5. Try to improve the performance by using other techniques (e.g. some advanced network embedding methods).
文中提到：GraphSGAN使用神经网络来捕捉特征之间的高阶相互关系。因此，尝试使用有监督的多层感知器的第一层输出重构图，并进一步观察到性能的改善，突出了神经网络在这个问题上的能力。
要进一步提升模型性能，第一种思路对模型进行提升。可以在参数选择上进一步细化；还可以用ensemble learning的思想，将文中作为baseline的方法与GraphSGAN结合，对神经网络进行结构改造，权衡各种因素的影响。
