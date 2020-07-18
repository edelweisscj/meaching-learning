# meaching-learning
Homework 6
Cai Jun
Read the paper (DeepInf: Social Influence Prediction with Deep Learning)

文章提出了一个基于深度学习的框架DeepInf，将影响动态和网络结构都表示为一个潜在空间。将网络嵌入、图卷积和图注意力机制构建到一个统一的框架中，为了预测一个用户v的动作状态，首先用RWR对她的本地邻居进行抽样。在获得局部网络之后，利用图卷积和注意力技术来学习潜在的预测信号。文章的实验结果表明，除了预先训练好的网络嵌入，不考虑任何手工制作的特性，使DeepInf成为一个“纯”的端到端学习框架。仍然可以获得相当的性能。

文章整理思维导图附在DeepInf_xmind.pdf
全文翻译整理在https://blog.csdn.net/weixin_42281282/article/details/107424423


图：DeepInf相关领域
Implement DeepInf-GCN and DeepInf-GAT to predict users’ retweeting behaviors on Weibo
Compare the prediction results with the results reported in the paper.

（1）参考https://github.com/xptree/DeepInf 代码利用作业微博数据集对文章进行了复现。参数选择基本参考论文。结果表明，GAT的性能明显优于GCN，但训练时间几乎是GCN双倍。复现结果与文中结果对比如下：
  Model		             AUC	Prec	 Rec	 F1
DeepInf-GCN	文章结果	76.85	42.44	71.30	53.21
	          复现结果	74.48	40.95	72.79	52.96
DeepInf-GAT	文章结果	82.72	48.53	76.09	59.27
	          复现结果	82.33	46.89	78.02	58.11
文章中有个很有趣的点，在和baseline（LR、SVM、PSCN）比较时，DeepInf-GCN在所有方法中性能最差，作者将其归因于homophily assumption of GCN，即相似的顶点比不相似的顶点更容易相互连接。在这样的假设下，对于一个特定的顶点，GCN通过取其邻域表示的未加权平均值来计算其隐藏表示。然而，在我们的应用中，同源性假设可能不成立。另一方面活跃邻居比不活跃邻居更重要，GAT使用图注意力来区别对待邻居，这也解释了其性能更优的原因。
故一般情况下性能优先下选用GAT。但若考虑时间经济性，可在性能条件满足、同源性假设成立前提下，可使用GCN节约时间成本。
（2）去除实例规范化层再一次训练，对比体会规范化的效果，可得到文中的结论。实例规范化层可显著地避免了过度拟合，使训练过程更加robust。

类比传统机器学习中数据预处理Normalization的z-score标准化，将不同特征维度的伸缩变换使得不同度量之间的特征具有可比性，可以大大提高训练器的准确性，在模型收敛过程中更加平缓robust，更容易正确收敛到最优解。
（3）预测性能如何随超参数的变化而变化也是很有趣的点。改变采样网络的大小，可以发现模型预测性能（根据AUC和F1）随着网格增大缓慢提高。因为随着采样网络规模的增加，我们获得了更多的信息。我们需要在计算复杂度和性能之间找到平衡点。另外，在 multi-head attention中平衡 head的个数和每个head的隐藏单元数目也是个需要设计的trade-off，以求找到最高性能点。复现时依据文中图3（d）选取第一层头的个数为4，文中为8，但是时间原因只跑了400个epochs，模型可能还没有达到最优，性能没有论文好。

improve the model’s performance by using other techniques

要进一步提升模型性能，有两种思路，两者也可以结合起来。
第一种思路对模型进行提升。可以在参数选择上进一步细化；还可以用ensemble learning的思想，将文中作为baseline的三个方法与两种DeepInf(DeepInf-GCN and DeepInf-GAT)结合，对神经网络进行结构改造，权衡各种因素的影响。
第二种思路是对数据部分做更多的工作，让数据本身更好用。文中最后提到通过利用强化学习，将采样和学习结合在一起。对采样和学习过程进行建模，借鉴DQN的搭建，抽象为智能体与环境学习交互的过程，把数据中更有用的信息提取出来。其中环境的搭建是难题。（程序中附上paddle框架下的RL.ipynb，是一次**unfinished**搭建尝试）
此外，个性化分析数据特征，在时间和空间两个层面上综合。文中没有考虑时间维度带来的变化信息量。利用原始数据时间截，结合新的混合跳传播层和扩展的起始层来捕获时间序列中的空间和时间相关性，将图学习、图卷积和时间卷积模块在DeepInf端到端框架中联合学习[1]。
（[1]参考https://arxiv.org/abs/2005.11650）
Last but not least
（1）本地跑巨慢，用Tesla K80上传数据包花费时间惊人，浪费大量使用时长，不知有什么好的方式？
（2）训练时间长，作业时间短，没有更多时间调参研究对比结果，调整方案性能提升，深感遗憾！
（3）文中讲DeepInf可以有效和高效地总结网络中的一个局域网。这些总结出来的表示可以应用到各种下游应用中，如链路预测、相似性搜索、网络对齐等。在我认为，6G端到端通信，用户行为预测必然关系到网络流量预测的变化，将DeepInf与强化学习结合，研究6G网络资源分配、星座星间与星地链路路由设计、多星多波束切换移动性管理等是很有意义的研究方向！从用户意图驱动输入到6G网络策略顶层设计形成闭环，真正契合6G 全球全域泛在human centric的愿景。
