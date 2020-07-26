## Semi-supervised Learning on Graphs with Generative Adversarial Nets

## 1. Read the paper.

文章研究了生成对抗网（generative attersarial nets，GANs）如何帮助图的半监督学习。

文章翻译整理在https://blog.csdn.net/weixin_42281282/article/details/107575772

#### Overview

提出了图的半监督学习的一种新方法GraphSGAN。在GraphSGAN中，生成器和分类器网络进行了一种新颖的竞争博弈。在平衡状态下，生成器在子图之间的低密度区域生成假样本。为了区分真假样本，分类器隐式地考虑了子图的密度特性。提出了一种有效的对抗学习算法，在理论上保证了对传统规范化图拉普拉斯正则化的改进。在几种不同类型的数据集上的实验结果表明，所提出的GraphSGAN明显优于几种最新的方法。GraphSGAN还可以使用小批量进行训练，因此具有可伸缩性优势。

#### Contribution

•引入了GANs作为解决半监督环境下图的分类问题的工具。GraphSGAN在图的低密度区域生成假样本，并利用聚类特性帮助分类。

•为GraphSGAN设计了一个新颖的鉴别器之间的竞争博弈，深入分析了训练过程中的dynamics、平衡和工作原理。此外，我们还对传统算法进行了改进。理论证明和实验验证都表明了该方法的有效性。

•在多个不同比例的数据集上评估我们的模型。GraphSGAN的性能明显优于以前的工作，并且展示了出色的可伸缩性。

#### Interesting part

使用GAN来估计密度子图，然后在密度空白区域生成样本。然后要求分类器先对假样本进行识别，然后再将其分类。这样，将假样本与真样本区分开来，会导致学习到的分类函数在密度间隙附近具有更高的曲率，从而削弱了穿过密度间隙传播的效果。同时，在每个子图内部，由于有监督的降损技术和一般的平滑技术，例如随机层，对正确标签的置信度将逐渐提高。

基于GAN的模型不能直接应用于图形数据。为此，GraphSGAN首先使用网络嵌入方法（例如，DeaveWalk等）来学习每个节点的潜在分布表示qi，然后将潜在分布qi与原始特征向量wi连接，即xi＝（wi，qi）。最后，xi作为我们的方法的输入。GraphSGAN中的分类器D和生成器G都是多层感知器。更具体地说，发生器以高斯噪声Z作为输入，并输出具有与席I形状相似的伪样本。在生成器中，使用批处理规范化。生成器的输出层受权重规范化技巧的约束，该技巧具有可训练的weight scale。GANs中的鉴别器由分类器代替，在输入后加入随机层（加性高斯噪声）和全连通层以达到平滑的目的。在预测模式下去除噪声。全连通层中的参数通过正则化的权重归一化来约束。分类器中最后一个隐藏层的输出是通过非线性变换从输入x中提取的特征，这对于训练生成器时的特征匹配至关重要。

归一化是降低边缘节点影响的核心。文中采用生成假节点的方法，将它们链接到最近的实节点，然后求解图的拉普拉斯正则化。假标签不允许分配给未标记的节点，损失计算只考虑真实节点之间的边。

## 2. Implement GraphSGAN on Cora dataset.
## 3. Compare your prediction results with the results reported in the paper.

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

(2)设置超参数平衡different losses的trade-off是很有意思的一个点。通过对超参数设置的改变研究不同loss权重对结果的影响，从而分析哪个条件中的满足对结果影响更为重要！（具体的loss与条件的对应分析见下一部分）


## 4. Analyze the effects of different losses.

GAN是一种通过对抗过程估计生成模型的新框架，其中生成模型G被训练成最适合原始训练数据，而判别模型D被训练来区分真实样本和模型G生成的样本。该过程可以形式化为一个介于D和D之间的最小-最大博弈。
在一个普通的博弈中，G和D有各自的损失函数，并试图使其最小化。他们的损失是相互依存的。文中表示为损耗函数L_G（G，D）和L_D（G，D）。效用函数V_G（G，D）和V_D（G，D）是负损失函数。GANs定义了一个零和博弈，其中L_G（G，D）=−L_D（G，D）。在这种情况下，唯一的纳什均衡可以通过极大极小策略来达到。
文中提出的GraphSGAN修改了L_D（G，D）和L_G（G，D）来设计一个新的博弈，在这个博弈中，G将在平衡点的密度间隙中生成样本。基于著名的“维度诅咒”，我们希望中心区域成为一个密度间隙而不是一个团簇。

定义 L_D = loss_supervised + lamda0 * loss_unsupervised + lamda1 * loss_ent + loss_pt

L_G = loss_fm + lamda2 * loss_pt

在均衡状态下，没有一个player可以改变他们的策略来单方面减少他的损失。假设G在平衡时在中心区域生成样本，文中提出了D在最后表示层（n维高维空间）中达到预期平衡的四个条件：

（1） 不同类的节点应该映射到不同的集群中。（2） 标记和未标记的节点都不应映射到中心区域，以使其成为一个密度间隙。（3） 每个未标记的节点都应该映射到一个表示特定标签的集群中。（4） 不同的集群应该足够远。

 loss_supervised is defined as the cross entropy between predicted distribution over M classes and one-hot representation for real label.所以loss_supervised的作用是用来满足条件（1），减小loss_supervised可使不同类的节点尽可能映射到不同的集群中。
 
 loss_unsupervised 根据最小-最大博弈的loss (value) function来定义。The classifier D incurs loss_unsupervised when real-or-fake misclassification happens.
介于G在中心密度间隙产生假样本，条件（2）相当于original GAN 中D的目的，所以 loss_unsupervised用来满足条件（2）。

loss_ent是M个标签上的分布熵（熵：概率分布不确定性的度量），是一个熵正则化项，用来满足条件（3），减少熵可以鼓励分类器为每个节点确定一个明确的标签。

loss_pt最初设计用于在GAN中产生不同的样本。它是批处理中向量之间的平均余弦距离，使最后表示层的表示尽可能远离其他层。条件（4）扩大了密度差距，有助于分类。因此，减小loss_pt可以满足条件（4），鼓励集群远离其他集群。

假设D满足上述四个条件，文中给了两个条件可以保证G在表示层中达到预期平衡：（a）G生成映射到中心区域的样本。（b） 生成的样本不应在唯一的中心点过度拟合。

产生性损失loss_gen用来满足这两个条件。对于条件（a），我们使用特征匹配损失loss_fm使生成样本与真实样本的中心点之间的距离最小化。实际训练过程中，中心点被一个真实的批处理席所取代，这也有助于满足条件（b）。条件（b）要求生成的样本覆盖尽可能多的中心区域。为了满足这个条件引入loss_pt(因为它鼓励产生不同的样本)。在中心性和多样性之间需要 trade-off，因此我们使用一个超参数λ2来平衡loss_fm和loss_pt。D中的随机层给假输入增加了噪声，这不仅提高了鲁棒性，而且防止了假样本的过拟合。

通过对lamda0、lamda1、lamda2 的取值分析，可以发现四个条件中满足条件（2）（ 标记和未标记的节点都不应映射到中心区域，以使其成为一个密度间隙）即减小loss_unsupervised更为重要；在中心性和多样性之间的 trade-off，减小loss_fm满足中心性是首要，其次考虑多样性防止过度拟合。



## 5. Try to improve the performance by using other techniques.


要进一步提升模型性能，有以下几种思路，可以综合采用。

（1）第一种思路对模型进行提升。

可以在参数选择上进一步细化，分析应满足条件为GraphSGAN设计更好的损失函数，研究更理想的平衡点；

还可以用ensemble learning的思想，将文中作为baseline的方法（regularization-based methods including LP , ICA  and ManiReg / embedding-based methods including DeepWalk, SemiEmb and Planetoid / convolution-based methods including Chebyshev , GCN and GAT ）与GraphSGAN结合，对神经网络进行结构改造，权衡各种因素的影响。

（2）第二种思路是对数据部分做更多的工作，优化数据采样。既要保证数据特征全面性，也要避免对网络结构信息的过度采样。

（3）根据任务特点寻求突破。文中在§4理论上证明了减少边缘节点的影响有助于分类。而归一化是降低边缘节点影响的核心。文中方法为生成假节点，将它们链接到最近的实节点，然后求解图的拉普拉斯正则化。假标签不允许分配给未标记的节点，损失计算只考虑真实节点之间的边。我们可以进一步寻求改进降低边缘节点影响的方法来提升分类性能。

（4）多种方式结合进一步稳定训练，加快训练速度，强化该方法的理论基础，并将该方法推广到其他图数据任务。

## 6. A little bit more...

文中提到：GraphSGAN使用神经网络来捕捉特征之间的高阶相互关系。因此，尝试使用有监督的多层感知器的第一层输出重构图，并进一步观察到性能的改善，突出了神经网络在这个问题上的能力。若对这个方法做迁移，是否会对很多高维数据的特征提取有很好的效果？基于特征对网络进行重构，是否可以有更好的性能？

图卷积的一个明显的缺点是占用大量空间，GraphSGAN克服了这个缺点。在ensemble提升性能时，可以针对性扬长避短。
