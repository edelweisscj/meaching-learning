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

因aws服务器总是无法跑通，故在本地Ubuntu下安装docker，尝试实现两种不需要trajectory dataset的算法(DDDQN与Rainbow)

https://github.com/minerllabs/baselines/tree/master/general/chainerrl 给出的baseline experimental results of DDDQN/Rainbow/PPO/BC/GAIL/DQfD

      use trajectory dataset?	Treechop	Navigate	NavigateDense
      
(paper) DDDQN	No	3.73 +- 0.61	0.00 +- 0.00	55.59 +- 11.38

(paper) A2C	No	2.61 +- 0.50	0.00 +- 0.00	-0.97 +- 3.32

(paper) BC	Yes	0.75 +- 0.39	4.23 +- 4.15	5.57 +- 6.00

(paper) PreDQN	Yes	4.16 +- 0.82	6.00 +- 4.65	94.96 +- 13.42

(ours) DDDQN	No	5.28 +- 2.87	4.0 +- 19.60	59.13 +- 52.43

(ours) Rainbow	No	62.44 +- 2.74	13.0 +- 33.63	66.89 +- 41.24

(ours) PPO	No	56.31 +- 8.31	8.0 +- 27.13	87.83 +- 59.46

(ours) BC	Yes	9.27 +- 5.21	46.00 +- 50.1	69.54 +- 57.02

(ours) GAIL	Yes	16.34 +- 6.85	32.00 +- 46.88	59.32 +- 30.60

(ours) DQfD	Yes	62.37 +- 2.16	6.00 +- 23.75	not evaluated

(paper) Human	-	64.00 +- 0.00	100.00 +- 0.00	164.00 +- 0.00

在实现过程中，利用了https://hub.docker.com/r/chenqibin422/minerl docker环境。由于硬件条件等各方面的约束，并没有得到好的训练效果，但能观察到reward不断提高，动作的奖励值是由开发者决定的，奖励值的设置好坏对学习效果有很大影响。

## For Competition.
竞赛要求The submissions must train a machine learning model without relying on human domain knowledge (no hardcoding, no manual specification of meta-actions e.g. move forward then dig down, etc). Participants can use the provided MineRL-v0 dataset of human demonstrations, but no external datasets.确实难度比较大，希望可以寻求队友共同参赛。

最近有几篇有意思的论文值得借鉴研究

1. What Can Learned Intrinsic Rewards Capture?

强化学习（Reinforcement Learning，RL）智能体的目标是奖励最大化。在此论文中，作者们认为奖励函数自身可以成为学习知识的好地方。为了进一步研究，他们提出了一个可伸缩的元梯度（meta-gradient）框架，跨多个生命周期学习有用的内在奖励函数，从而表明，学习并捕获有关长期探索和开发的知识到奖励函数是可行的。
参考链接：https://analyticsindiamag.com/papers-icml-2020-research-conference/

2. Discovering Reinforcement Learning Algorithms

Alphabet 旗下的 DeepMind ，正在寻找新的方法来进一步提高算法自主学习的自动化程度：让算法自己处理顶级计算机科学家可能都要花好几年时间才能完成的复杂编程任务。
在预印本网站 arXiv 上发表的一篇最新论文中，DeepMind 团队描述了一种新的深度强化学习算法，该算法能够发现其自身的值函数（value function）
## A little bit more...

• 简单来讲，强化学习的主要目的是研究并解决智能体或多智能体贯序决策的问题，在理论与我专业领域的自动控制有共通之处，很多人说强化学习是人工智能的未来，因为它更接近生物自然学习的过程。然而机器是否真正理解交互过程的真正含义？尽管如今深度强化学习使得效果大幅提升，在视频游戏、棋类游戏、资源调配、自动调参等领域大显神通，然而复杂环境的建立难度，奖励值的设置人为影响，值函数近似误差带来的过估计问题等……这些仍然使得当前的强化学习被局限在很小的应用范围内，无法解决复杂环境中的实际决策问题。强化学习是否还需要原理性的突破才能面对未来各项应用挑战？未来是否进一步与知识图谱等结合，【知识驱动+环境驱动+数据驱动】，融合领域所长，促进AI进一步发展？

• 想做好AI，硬件和软件算法同样重要，迫切想有个好用的服务器，可以无障碍跑程序。下一步就是花经费去，争取近期把自己的服务器建起来。

• 努力提升自身各方面能力，包括代码能力，硬件配置调试能力，钻研算法，力争在理论上有所突破，出高水平论文。

• 从课程中get到了不少新知识和新技能，非常感谢唐老师和各位助教老师以及课程建设人员/组织者的辛苦付出！！学无止境！！人工智能是一个多学科交叉的闪光点，潜力无限！我会结合本专业所长在这个跨学科领域继续耕耘下去，并把收获带给自己的学生，期望AI的未来有我们奋斗的痕迹~~


## References
[1] NeurIPS 2019 Competition: The MineRL Competition on Sample Efficient Reinforcement Learning using Human Priors
[2] Playing Atari with Deep Reinforcement Learning
[3] Deep Reinforcement Learning with Double Q-learning
[4] Prioritized Experience Replay
[5] Dueling Network Architectures for Deep Reinforcement Learning
[6] Reinforcement Learning: An Introduction
[7] A Distributional Perspective on Reinforcement Learning
[8] Rainbow: Combining Improvements in Deep Reinforcement Learning
[9] 白话强化学习与Pytorch
[10] What Can Learned Intrinsic Rewards Capture?
[11] Discovering Reinforcement Learning Algorithms

