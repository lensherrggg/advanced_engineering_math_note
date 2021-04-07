# 数理统计

高等工程数学笔记

## 几种重要随机变量的数学期望与方差
### 二点分布
若随机变量$X$服从二点分布，其分布律为：
| $X$ | 0 | 1 |
| ---- | ---- | ----|
| $P_k$ | 1 - p | p |

$ E(X) = p $
$ E(X^2) = p $
$ D(X) = p(1-p) $

### 二项分布
随机变量$X \sim B(n,p)$，其分布律为
$$ P\{ X=k \} = C_n^kp^k(1-p)^{n-k}, \quad k=1,2,\cdots,n $$

$ E(X) = np $
$ D(X) = np(1-p) $

### 泊松分布

随机变量$X \sim P(\lambda) $，其分布律为
$$
P\{ X=k \}=\frac{\lambda ^ke^{-\lambda}}{k!}, \quad k=0,1,2,\cdots
$$

$ E(X) = \lambda $
$ D(X) = \lambda $

泊松分布表示单位时间内发生n次某事件的概率

### 均匀分布

设随机变量$X$在区间$(a,b)$上服从均匀分布，$X\sim U(a,b)$，其概率密度为
$$
f(x)=\begin{cases}
\frac{1}{b-a} \quad &a < x < b, \\
0 &其他
\end{cases}
$$

$ E(X) = \frac{a+b}{2} $
$ D(X) = \frac{(b-a)^2}{12} $

### 正态分布
随机变量$X\sim N(\mu, \sigma)$，其概率密度为
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \quad - \infty < x < + \infty
$$

$ E(X) = \mu $
$ D(X) = \sigma^2 $

### 指数分布
随机变量X服从参数为$\lambda$的指数分布，$ X\sim E(\lambda) $，其概率密度为
$$
f(x)=
\begin{cases}
\lambda e^{-\lambda x} \quad &x > 0 \\
0 &x \le 0
\end{cases}
$$

$ E(X) = \frac{1}{\lambda} $
$ D(X) = \frac{1}{\lambda^2} $

指数分布表示一件事两次发生的时间间隔的分布

## 数理统计基本概念

### 总体和个体
研究对象的全体称为**总体**

组成总体的每个元素称为**个体**

### 样本

为推断总体分布及各种特征，按一定规则从总体中抽取若干个体进行观察试验，以获得有关总体的信息，这一抽取过程称为 “**抽样**”，所抽取的部分个体称为**样本**. 样本中所包含的个体数目称为**样本容量**

样本的抽取是随机的，每个个体是一个随机变量，容量为n的样本可看做n维随机变量，用$X_1, X_2, \cdots, X_n$表示

一旦选定一组样本，得到的是n个具体的数$(x_1, x_2, \cdots, x_n)$，称其为样本的一个观察值，简称样本值。

## 统计量与抽样分布

### 统计量

**定义**：若样本$X_1, X_2, \cdots, X_n$的函数$g(X_1, X_2, \cdots, X_n)$中不含有任何未知参数，则称函数$g(X_1, X_2, \cdots, X_n)$为统计量
若$x_1, x_2, \cdots, x_n$是相应的样本值，则称函数值$g(x_1, x_2, \cdots, x_n)$为统计量$g(X_1, X_2, \cdots, X_n)$的一个观察值。

#### 几个常用统计量

##### 样本均值

$$
\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

##### 样本方差

$$
S^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2
$$

它反映了总体方差的信息

##### 样本$k$阶原点矩

$$
A_k=\frac{1}{n}\sum_{i=1}^{n}X_i^k
$$

它反映总体k阶矩的信息

##### 样本$k$阶中心矩

$$
B_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k
$$

它反映了总体k阶中心矩的信息

它们的观察值分别为

$$
\overline{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\overline{x})^2
$$

$$
a_k=\frac{1}{n}\sum_{i=1}^{n}x_i^k
$$

$$
b_k=\frac{1}{n}\sum_{i=1}^n(x_i-\overline{x})^k
$$

由大数定律可知

$$
A_k=\frac{1}{n}\sum_{i=1}^{n}X_i^k
$$
依概率收敛于$E(X^k)$

## 统计学中三个常用分布和上$\alpha$分位点

### 抽样分布

统计量是样本的函数，而样本是随机变量，故统计量也是随机变量，因而就有一定的分布，它的分布称为“**抽样分布**”

#### $\mathcal{X}^2$分布

**定义**：设$X_1, X_2, \cdots, X_n$相互独立，都服从$N(0,1)$，则称随机变量
$$\mathcal{X}^2=X_1^2+X_2^2+\cdots +X_n^2$$所服从的分布为自由度为n的$\mathcal{X}^2$分布，记为$$\mathcal{X}^2\sim \mathcal{X}^2(n)$$

$\mathcal{X}^2$分布的概率密度图形如下

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210405214442.png)

显然$\mathcal{X}^2$分布的概率密度图形随自由度的不同而有所改变

#### 性质

**性质1**：设$\mathcal{X}^2\sim \mathcal{X}^2(n)$，则$E(\mathcal{X}^2)=n,D(\mathcal{X}^2)=2n$

**性质2**：设$\mathcal{X}_1^2\sim \mathcal{X}^2(n_1)$，$\mathcal{X}_2^2\sim \mathcal{X}^2(n_2)$，且$\mathcal{X}_1^2$与$\mathcal{X}_2^2$相互独立，则$$\mathcal{X}_1^2+\mathcal{X}_2^2\sim \mathcal{X}^2(n_1+n_2)$$

这称为$\mathcal{X}^2\$分布的可加性

#### $t$分布

**定义**：设$X\sim N(0,1), Y\sim \mathcal{X}^2(n)$，且$X$与$Y$相互独立，则称变量$t=\frac{X}{\sqrt{Y/N}}$所服从的分布为自由度n的t分布，记为$t\sim t(n)$

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210405215012.png)

t分布的概率密度函数关于t=0对称，且当n充分大时$n \ge 30$，其图形与标准正态分布的 概率密度函数的图形非常接近.但对于较小的n，t分布与$N\sim(0,1)$分布相差很大.

#### $F$分布

**定义**：设$X\sim \mathcal{X}^2(n_1), Y\sim \mathcal{X}^2(n_2)$，$X$与$Y$相互独立，则称统计量$F=\frac{X/n_1}{Y/n_2}$服从自由度为$n_1$和$n_2$的$F$分布，$n_1$称为第一自由度，$n_2$称为第二自由度，记作$F\sim F(n_1, n_2)$。

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210405215332.png)

#### 上$\alpha$分位点

**定义**：设随机变量$X$的概率密度为$f(x)$，对于任意给定的$\alpha(0 < \alpha < 1)$，若存在实数$x_{\alpha}$使得$$ P\{ X \ge x_{\alpha} \} = \int_{x_\alpha}^{+\infty}f(x)dx=\alpha$$则称$x_\alpha$为概率分布的上$\alpha$分位点。

##### 正态分布

$$ P\{ Z\ge z_{\alpha} \} = \int_{z_\alpha}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt=\alpha$$

即$P\{ Z < z_\alpha \}=1-\alpha$，$\Phi(z_\alpha)=1-\alpha$确定点$z_\alpha$

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210405220113.png)

#### $\mathcal{X}^2$分布

当n充分大时$(n > 45)$

$$\mathcal{X}^2 \approx \frac{1}{2}(Z_\alpha + \sqrt{2n-1})^2$$

其中$Z_\alpha$是标准正态分布的上$\alpha$分位点

#### $t$分布

1. 由于其对称性，有：$t_{1-\alpha}(n)=-t_\alpha(n)$
2. 当n充分大时$(n > 45)$，$t_\alpha(n)=Z_\alpha$

#### $F$分布

$$F_{1-\alpha}(n_1,n_2)=\frac{1}{F_\alpha(n_2, n_1)}$$

## 抽样分布定理

**定理1**：设$X_1,X_2,\cdots,X_n$是取自正态总体$N\sim (\mu, \sigma^2)$的样本，则有
1. 样本均值$\overline{X}\sim N(\mu, \frac{\sigma^2}{n})$
2. 样本均值$\overline{X}$与样本方差$S^2$相互独立
3. 随机变量$$\frac{(n-1)S^2}{\sigma^2}=\frac{\sum_{i=1}^n(X_i-\overline{X})^2}{\sigma^2}\sim\mathcal{X}^2(n-1)$$

**定理2**：设$X_1,X_2,\cdots,X_n$是取自正态总体$N(\mu, \sigma^2)$的样本，$\overline{X}$和$S^2$分别为样本均值和样本方差，则有$$\frac{\overline{X}-\mu}{S/\sqrt{n}}\sim t(n-1)$$

**定理3（两个总体样本均值差的分布）**：设$X\sim N(\mu_1,\sigma^2)$,$Y\sim N(\mu_2, \sigma^2)$且$X$与$Y$独立，$X_1, X_2, \cdots, X_{n_1}$是取自$X$的样本，$Y_1, Y_2, \cdots, Y_{n_2}$是取自$Y$的样本，$\overline{X}$和$\overline{Y}$分别是这两个样本的样本均值，$S_1^2$和$S_2^2$分别是这两个样本的样本方差，则有$$\frac{\overline{X}-\overline{Y}-(\mu_1-\mu_2)}{\sqrt{\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}}\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}\sim t(n_1+n_2-2)$$

**定理4（两个总体样本方差比的分布）**：$X\sim N(\mu_1,\sigma_1^2)$,$Y\sim N(\mu_2,\sigma_2^2)$且$X$与$Y$独立，$X_1, X_2, \cdots, X_{n_1}$是取自$X$的样本，$Y_1, Y_2, \cdots, Y_{n_2}$是取自$Y$的样本，$\overline{X}$和$\overline{Y}$分别是这两个样本的样本均值，$S_1^2$和$S_2^2$分别是这两个样本的样本方差，则有$$\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F(n_1-1,n_2-1)$$

## 大数定律

**定义**：设$Y_1, Y_2, \cdots, Y_n$为一个随机变量序列，$a$是一个常数，若对任一正数$\epsilon > 0$，总成立$$\lim_{n->\infty}P\{|Y_n-a| < \epsilon\} = 1$$

则称随机变量序列$Y_1, Y_2, \cdots, Y_n$依概率收敛于$a$，记为$$Y_n \stackrel{P}{\rightarrow} a(n\rightarrow \infty)$$

### 性质

1. 设$Y_n \stackrel{P}{\rightarrow} a(n\rightarrow \infty)$，$g(x)$是连续函数，则$$g(Y_n) \stackrel{P}{\rightarrow} g(a) (n\rightarrow \infty)$$
2. 设$X_n \stackrel{P}{\rightarrow} a(n\rightarrow \infty), Y_n \stackrel{P}{\rightarrow} b(n\rightarrow \infty), g(x, y)$是二元连续函数，则$$g(X_n, Y_n) \stackrel{P}{\rightarrow} g(a, b) (n\rightarrow \infty)$$

### 三个常见的大数定律

#### 伯努利大数定律

设n重伯努利试验中事件A发生的次数为$\mu_n$，A在每次实验中发生的概率为$p$，则对任给的$\epsilon > 0$，总成立$$\lim_{n\rightarrow \infty}P\{|\frac{\mu_n}{n}-p| < \epsilon\} = 1$$

即$$\frac{\mu_n}{n}\stackrel{P}{\rightarrow}p(n\rightarrow \infty)$$

##### 伯努利大数定律的意义

在概率的统计意义中，事件A发生的频率$\frac{n_A}{n}$稳定于p

#### 切比雪夫大数定律的特殊情形

设随机变量序列$X_1, X_2, \cdots $相互独立，并且具有相同的数学期望和方差，$E(X_i)=\mu,D(X_i)=\sigma^2, i=1, 2, \cdots$，则对任给的$\epsilon > 0$总成立$$\lim_{n\rightarrow \infty}P\{|\frac{1}{n}\sum_{i=1}^nX_i-\mu|<\epsilon\}=1$$

即$$\frac{1}{n}\sum_{i=1}^nX_i\stackrel{P}{\rightarrow}\mu(n\rightarrow \infty)$$

##### 定理的意义

具有相同数学期望和方差的独立随机变量序列的算术平均值依概率收敛于数学期望.当n足够大时, 实验结果的算术平均几乎是一常数

#### 切比雪夫大数定律的一般情形

设随机变量序列$X_1, X_2, \cdots $相互独立，并且具有数学期望$E(X_i)=\mu_i$,并且都具有被同一常数$C$限制的方差$D(X_i)=\sigma^2 < C, i=1, 2, \cdots$，则对任给的$\epsilon > 0$总成立$$\lim_{n\rightarrow \infty}P\{|\frac{1}{n}\sum_{i=1}^nX_i-\frac{1}{n}\sum_{i=1}^{n}\mu_i|<\epsilon\}=1$$

即$$\frac{1}{n}\sum_{i=1}^nX_i\stackrel{P}{\rightarrow}\frac{1}{n}\sum_{i=1}^n\mu_i(n\rightarrow \infty)$$

##### 意义

定理表明，独立随机变量序列$\{X_n\}$，若方差有共同的上界，则$\frac{1}{n}\sum_{i=1}^nX_i$与数学期望$\frac{1}{n}\sum_{i=1}^{n}E(X_i)$偏差很小的概率接近1

#### 辛钦大数定律

设随机变量序列$X_1, X_2, \cdots $相互独立，服从同一分布，具有相同的数学期望$E(X_i)=\mu, i=1,2,\cdots$,则对于任给正数$\epsilon > 0$，总成立$$\lim_{n\rightarrow \infty}P\{|\frac{1}{n}\sum_{i=1}^{n}X_i-\mu|<\epsilon\}=1$$

**推论**：设随机变量序列$X_1, X_2, \cdots $相互独立，服从同一分布，具有相同的k阶矩$E(X_i^k)=\mu_k, i=1,2,\cdots$,则对于任给正数$\epsilon > 0$，总成立$$\lim_{n\rightarrow \infty}P\{|\frac{1}{n}\sum_{i=1}^{n}X_i^k-\mu_k|<\epsilon\}=1$$

大数定律表达了随机现象最根本的性质之一：平均结果的稳定性

## 中心极限定理

客观实际中，许多随机变量是由大量 相互独立的偶然因素的综合影响所形成，每一个微小因素，在总的影响中所起的作用是很小的，但总起来，却对总和有显著影响，这种随机变量往往近似地服从正态分布。

由于无穷个随机变量之和可能趋于$\infty$，故我们不研究n个随机变量之和本身而考虑它的标准化的随机变量$$Z_n=\frac{\sum_{k=1}^nX_k-E(\sum_{k=1}^nX_k)}{\sqrt{D(\sum_{k=1}^nX_k})}$$

的极限分布

### 定理1（独立同分布下的中心极限定理）

设$X_1,X_2,\cdots$是独立同分布的随机变量序列，且$E(X_i)=\mu, D(X_i)=\sigma^2,i=1,2,\cdots$则$$\lim_{n\rightarrow \infty}P\{\frac{\sum_{i=1}{n}X_i-n\mu}{\sigma\sqrt{n}}\le x\}=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt$$

**定理表明**：当n充分大时，标准化随机变量$\frac{\sum_{i=1}{n}X_i-n\mu}{\sigma\sqrt{n}}$近似服从正态分布。

由此可知，对独立随机变量序列，不管服从什么分布，只要是同分布且有有限的期望和方差，则n充分大时随机变量之和$\sum_{i=1}^nX_i$近似服从正态分布$N(n\mu,n\sigma^2)$

### 定理2（德莫佛——拉普拉斯中心极限定理）

设n重伯努利试验中事件A发生的次数为$\mu_n$，事件A在每次实验中发生的概率为$p$，则对于任给实数$x$总成立$$\lim_{n\rightarrow \infty}P\{\frac{\mu_n-np}{\sqrt{np(1-p)}}\le x\}= \int_{-\infty}^x\frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt$$

定理表明：若$Y$服从二项分布，当n很大时，$Y_n$的标准化随机变量$\frac{Y_n-np}{\sqrt{np(1-p)}}$近似服从标准正态分布。

由此可知，当n很大，$0<p<1$是一个定值时，服从二项分布$B(n,p)$的随机变量$Y_n$近似服从正态分布$N(np,np(1-p))$

### 定理3（李雅普诺夫中心极限定理）

设$X_1,\cdots,X_n,\cdots$相互独立，且$EX_k=\mu_k,DX_k=\sigma_k^2,(k=1,2,\cdots)$，记$B_n^2=\sum_{k=1}^{n}\sigma_k^2$，若存在正数$\delta$使得当$n\rightarrow \infty$时，$\frac{1}{B_n^{2+\delta}}\sum_{k=1}^nE\{|X_k-\mu_k|^{2+\delta\}}\rightarrow 0$,则$$\lim_{n\rightarrow \infty}P\{\frac{\sum_{k=1}^{n}X_k-\sum_{k=1}^n\mu_k}{B_n}\le x\}=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{-\frac{t^2}{2}}dt$$

## 参数估计的意义和种类

to be continued

## 估计量的评选标准

### 无偏性

**定义**：设$\hat{\theta}$是未知参数$\theta$的估计量，若$E(\hat{\theta})=\theta$则称$\hat{\theta}$是$\theta$的无偏估计量

### 有效性

**定义**：设$\hat{\theta_1}=\theta_1(X_1, X_2, \cdots, X_n)$和$\hat{\theta_2}=\theta_2(X_1, X_2, \cdots, X_n)$都是总体参数$\theta$的无偏估计量，且$$D(\hat{\theta_1}) < D(\hat{\theta_2})$$
则称$\hat{\theta_1}$比$\hat{\theta_2}$更有效

#### 罗—克拉美（Rao-Cramer）不等式

若$\hat{\theta}$是参数$\theta$的无偏估计量，则$$D(\hat{theta}) \ge \frac{1}{nE[\frac{\partial}{\partial\theta}ln \  p(X,\theta)]^2}=D_0(\theta)$$

其中$p(x,\theta)$是总体$X$的分布律或概率密度，称$D_0(\theta)$为方差的下界。

当$D(\hat{\theta})=D_0(\theta)$时，称$\hat{\theta}$为$\theta$的达到方差下界的无偏估计量，此时称$\hat{\theta}$为最有效的估计量，简称有效估计量

### 一致性

**定义**：设$\hat{\theta}=\theta(X_1, X_2, \cdots, X_n)$是总体参数$\theta$的估计量。若对于任意的$\theta \in \Theta$，当$n\to \infty$时，$\hat{\theta}$以概率收敛于$\theta$，即对于任意正数$\epsilon$有$\lim\limits_{n\to \infty}P(|\hat{\theta}-\theta|\ge \epsilon)=0$，则称$\hat{\theta}$是总体参数$\theta$的一致估计量

#### 关于一致性的两个常用结论

1. 样本$k$阶矩是总体$k$阶矩的一致估计量（由大数定律证明）
2. 设$\hat{\theta}$是$\theta$的无偏估计量且$\lim\limits_{n\to \infty}D(\hat{\theta})=0$，则$\hat{\theta}$是$\theta$的一致估计量（由切比雪夫不等式证明）

一般，矩估计法得到的估计量为一致估计量
