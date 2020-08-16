## Participate in MineRL 
https://www.aicrowd.com/challenges/neurips-2020-minerl-competition 

The MineRL 2020 Competition aims to foster the development of algorithms which can efficiently leverage human demonstrations to drastically reduce the number of samples needed to solve complex, hierarchical, and sparse environments. 

Participants will submit agents (round 1) and the code to train them from scratch (round 2) to AICrowd. Agents must train in under 8,000,000 samples from the environment on at most 1 P100 GPU for at most 4 days of training. The submissions must train a machine learning model without relying on human domain knowledge (no hardcoding, no manual specification of meta-actions e.g. move forward then dig down, etc). Participants can use the provided MineRL-v0 dataset of human demonstrations, but no external datasets.

## Environment setting & baselines Implementing

参考https://github.com/minerllabs/minerl 和 https://github.com/minerllabs/competition_submission_template 配置环境

baselines中包含以下几种
[baselines/dddqn.sh]
Double Dueling DQN (DDDQN)
[baselines/rainbow.sh]
Rainbow
[baselines/ppo.sh]
PPO
[baselines/behavoral_cloning.sh]
Behavoral Cloning (BC)
[baselines/gail.sh]
GAIL
[baselines/dqfd.sh]
DQfD

因aws服务器总是无法跑通，故在本地Ubuntu下安装docker，尝试实现不需要trajectory dataset的算法(DDDQN与Rainbow)

#### Overview

文章提出了一种新的面向web文档多跳问答的CogQA框架。该框架以认知科学中的双过程理论为基础，通过协调隐式提取模块（系统1）和显式推理模块（系统2）（系统1从段落中提取与问题相关的实体和答案，并对其语义信息进行编码。提取的实体被组织成一个认知图，类似于工作记忆。然后系统2在图上执行推理过程，并收集线索以指导系统1更好地提取下一跳实体），在迭代过程中逐步构建认知图。在给出准确答案的同时，框架还提供了可解释的推理路径。
#### Contribution

• 文中提出了一种新的基于人类认知的多跳阅读理解问答框架。

• 文章表明，我们框架中的认知图结构提供了有序和整体的可解释性，适合于关系推理。

• 文章基于BERT和GNN的实现在所有指标上大大超过了以前的工作和其他竞争对手。

#### interest

• 

• 

• 

## Reproduce CogQA results.




## Improve CogQA.
利用tune库超参数调优，
System1和System2进一步融合，模拟大脑思维的过程，增加系统间的通道数，优化系统之间的交互。

System1 进一步优化网络结构，结合注意力和递归机制的未来架构提高容量。

System2 借鉴MDO（多学科优化）思想、
AutoML、
元学习等，进一步完善可解释性和推理逻辑，利用神经逻辑技术来提高可靠性。
强化学习（Reinforcement Learning，RL）智能体的目标是奖励最大化。在此论文中，作者们认为奖励函数自身可以成为学习知识的好地方。为了进一步研究，他们提出了一个可伸缩的元梯度（meta-gradient）框架，跨多个生命周期学习有用的内在奖励函数，从而表明，学习并捕获有关长期探索和开发的知识到奖励函数是可行的。

参考链接：
https://analyticsindiamag.com/papers-icml-2020-research-conference/

## A little bit more...

• 

• 此外尝试改进程序with SAVING AND LOADING MODELS，以求克服12小时top的服务器运行时间。

• redis和 tmux 好玩好用，新技能get!

• 文中提到：框架可以推广到其他认知任务，例如会话人工智能和顺序推荐。我认为还可以适用于很多其他的任务。比如意图驱动，与可穿戴设备融合，替代人的认知和推理————数据由我、随心而动。

• 学科交叉，从Systerm1到System2，AI从感知走向认知，追求可解释性，意味着“感性”到“理性”的飞跃。虽然目前我们依旧很懵懂，很多问题待解决，我们似乎窥到了一点点AI future的模样，未来可期！

## References
[1] NeurIPS 2019 Competition: The MineRL Competition on Sample Efficient Reinforcement Learning using Human Priors
[2] Playing Atari with Deep Reinforcement Learning
[3] Deep Reinforcement Learning with Double Q-learning
[4] Prioritized Experience Replay
[5] Dueling Network Architectures for Deep Reinforcement Learning
[6] Reinforcement Learning: An Introduction
[7] A Distributional Perspective on Reinforcement Learning
[8] Rainbow: Combining Improvements in Deep Reinforcement Learning
[9] The MineRL Competition on Sample Efficient Reinforcement Learning using Human Priors

