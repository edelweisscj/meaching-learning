## Participate in MineRL 
https://www.aicrowd.com/challenges/neurips-2020-minerl-competition 

The MineRL 2020 Competition aims to foster the development of algorithms which can efficiently leverage human demonstrations to drastically reduce the number of samples needed to solve complex, hierarchical, and sparse environments. 

## Read the influential HotpotQA dataset 
https://arxiv.org/pdf/1809.09600.pdf

Cognitive Graph for Multi-Hop Reading Comprehension at Scale

文章翻译整理在https://blog.csdn.net/weixin_42281282/article/details/107453137

#### Overview

文章提出了一种新的面向web文档多跳问答的CogQA框架。该框架以认知科学中的双过程理论为基础，通过协调隐式提取模块（系统1）和显式推理模块（系统2）（系统1从段落中提取与问题相关的实体和答案，并对其语义信息进行编码。提取的实体被组织成一个认知图，类似于工作记忆。然后系统2在图上执行推理过程，并收集线索以指导系统1更好地提取下一跳实体），在迭代过程中逐步构建认知图。在给出准确答案的同时，框架还提供了可解释的推理路径。
#### Contribution

• 文中提出了一种新的基于人类认知的多跳阅读理解问答框架。

• 文章表明，我们框架中的认知图结构提供了有序和整体的可解释性，适合于关系推理。

• 文章基于BERT和GNN的实现在所有指标上大大超过了以前的工作和其他竞争对手。

#### interest

• 要跨越机器与人类阅读理解能力的鸿沟，还有三个主要挑战摆在面前：
1）推理能力。单段问答模型倾向于在与问题匹配的句子中寻找答案，这并不涉及复杂的推理。因此，多跳QA成为下一个需要攻克的前沿。
2） 可解释性。显式推理路径能够验证逻辑的严密性，对于QA系统的可靠性至关重要。HotpotQA要求模型提供支持句子，这意味着无序和句子级的解释能力，然而人类可以用一步一步的解决方案来解释答案，这表明了一种有序和整体性的解释能力。
3） 可扩展性。对于任何实用的QA系统，可伸缩性是必不可少的。现有的基于机器理解的问答系统一般遵循DrQA中的检索抽取框架，通过预检索将源的范围缩小到几个段落。与人类在海量内存中通过知识进行推理的能力相比，该框架是单段问答和可伸缩信息检索之间的简单折衷。

• 经验告诉我们：对这些挑战的解决方案的见解可以从人类的认知过程中获得。双过程理论（Evans，198420032008；Sloman，1996）表明，我们的大脑首先通过一个被称为系统1的隐式、无意识和直觉的过程，即系统1，在这个过程的基础上进行另一个显式、有意识和可控的推理过程，即系统2，来检索相关信息。系统1可以根据请求提供资源，而系统2可以通过在工作记忆中执行顺序思维来深入研究关系信息，后者速度较慢，但具有人类独特的理性。对于复杂的推理，这两个系统是协调的，以执行快速和缓慢的思考迭代。

• 学科的交叉融合，基础科学的理论迁移与技术转化在此体现。

## Reproduce CogQA results.

参考 https://github.com/THUDM/CogQA https://github.com/qibinc/CogQA 和https://github.com/dogydev/CogQA 复现论文

运行结果
{'em': 0.31427413909520594, 'f1': 0.452285260928425, 'prec': 0.50057944546263254, 'recall': 0.49101315200975601, 'sp_em': 0.227022282241728562, 'sp_f1': 0.5664633505889512, 'sp_prec': 0.6250258834121089, 'sp_recall': 0.5339292948779767, 'joint_em': 0.102835921674544227, 'joint_f1': 0.32569627417348081, 'joint_prec': 0.37554205116662915, 'joint_recall': 0.337765240089584142}

文章结果
{'em': 0.37555705604321404, 'f1': 0.49404898995339086, 'prec': 0.522077693107403, 'recall': 0.49920254582019247, 'sp_em': 0.23119513841998648, 'sp_f1': 0.5852652666167758, 'sp_prec': 0.642654924975991, 'sp_recall': 0.5968465644191497, 'joint_em': 0.12180958811613775, 'joint_f1': 0.3528686167056055, 'joint_prec': 0.4028774305383, 'joint_recall': 0.3646712724258193}

## Improve CogQA.
利用tune库超参数调优，
System1和System2进一步融合，模拟大脑思维的过程，增加系统间的通道数，优化系统之间的交互。

System1 进一步优化网络结构，结合注意力和递归机制的未来架构提高容量。

System2 借鉴MDO（多学科优化）思想、
AutoML、
元学习等，进一步完善可解释性和推理逻辑，利用神经逻辑技术来提高可靠性。

## A little bit more...

• 迄今为止花费最多时间在配置和训练上的project，GPU的弊端在于 memory 不够大，提供的服务器只有1个GPU，System2的batch-size直接降到1才勉强跑，冗长的训练时间带来低效的improve效率。尝试修改把部分模型放在本地CPU上跑（参考 https://github.com/THUDM/CogQA/issues/18 ），避免cuda out of memory 带来的训练漏数据的问题，提升训练模型回答问题性能。

• 此外尝试改进程序with SAVING AND LOADING MODELS，以求克服12小时top的服务器运行时间。

• redis和 tmux 好玩好用，新技能get!

• 文中提到：框架可以推广到其他认知任务，例如会话人工智能和顺序推荐。我认为还可以适用于很多其他的任务。比如意图驱动，与可穿戴设备融合，替代人的认知和推理————数据由我、随心而动。

• 学科交叉，从Systerm1到System2，AI从感知走向认知，追求可解释性，意味着“感性”到“理性”的飞跃。虽然目前我们依旧很懵懂，很多问题待解决，我们似乎窥到了一点点AI future的模样，未来可期！

