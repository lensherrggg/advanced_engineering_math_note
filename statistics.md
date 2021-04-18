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

这称为$\mathcal{X}^2$分布的可加性

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

### 参数估计问题

数理统计的基本问题是根据样本提供的信息对总体的分布及分布的某些数字特征做推断。这个问题中的一类是总体分布的类型已知，而它的某些参数未知，根据所得样本对这些参数做推断，这类问题称为参数估计。

### 未知参数的估计量和估计值

设一个总体X，其分布函数$F(x, \theta)$，其中$\theta$为未知参数($\theta$也可以是未知向量）。现在从该总体抽样，得到样本$X_1, X_2, \cdots, X_n$，样本值$x_1, x_2, \cdots, x_n$

若构造出适当的统计量$g(X_1, X_2, \cdots, X_n)$来估计$\theta$，则称$g(X_1, X_2, \cdots, X_n)$为$\theta$的估计量，将样本值$x_1, x_2, \cdots, x_n$代入，则称$g(x_1, x_2, \cdots, x_n)$为$\theta$的估计值。

### 参数估计的种类

+ 点估计：估计未知参数的值
+ 区间估计：估计未知参数的取值范围，并使此范围包含未知参数真值的概率为给定的值。

## 点估计的求法

### 矩估计法

理论依据：辛钦大数定律及其推论

方法：用样本k阶矩$A_k=\frac{1}{n}\sum_{i=1}^nX_i^k$估计总体k阶矩$\mu_k=E(X^k)$，建立含有待估参数的方程，从而解出待估参数。

步骤：设总体的分布函数的形式已知，待估参数为$\theta_1, \theta_2, \cdots, \theta_k$，总体的前k阶矩存在

1. 求总体的前k阶矩，一般是这k个参数的函数记为$$E(X^r)=\mu_r(\theta_1, \theta_2, \cdots, \theta_k), r=1, 2, \cdots, k$$样本$X_1, X_2, \cdots, X_n$的前k阶矩记为$$A_r=\frac{1}{m}\sum_{i=1}^{n}X_i^r,r=1,2,\cdots, k$$
2. 令$\mu_r(\theta_1, \theta_2, \cdots, \theta_k)=\frac{1}{n}\sum_{i=1}^nX_i^r,r=1,2,\cdots, k$，这是含未知参数$\theta_1, \theta_2, \cdots, \theta_k$的k个方程构成的方程组
3. 解这个方程组，得到k个统计量称为未知参数$\theta_1, \theta_2, \cdots, \theta_k$的矩估计量，代入样本值得到k个数称为矩估计值。

### 极大似然估计法

理论依据：极大似然原理

一般说，若时间A发生的概率与参数$\theta \in \Theta$有关，$\theta$取之不同，P(A)也不同。则应记事件A发生的概率为$P(A|\theta)$。若一次实验事件A发生了，可认为此时的$\theta$值应当是$\Theta$中使$P(A|\theta)$达到最大的那个，这就是极大似然原理。

似然函数：$X_1, X_2, \cdots, X_n$是取自总体X的样本，$x_1, x_2, \cdots, x_n$是样本值

1. X是离散型总体，其分布律为$$P\{X=x\}=p(x, \theta_1, \theta_2, \cdots, \theta_k)$$,其中$\theta_1, \theta_2, \cdots, \theta_k$为为止待估参数，则样本的联合分布律为
  $$
  \begin{aligned}
  P\{X_1=x_1, X_2=x_2, \cdots, X_n=x_n\}
  &=p(x_1, \theta_1, \theta_2, \cdots, \theta_k)p(x_2, \theta_1, \theta_2, \cdots, \theta_k)\cdots p(x_n, \theta_1, \theta_2, \cdots, \theta_k) \\&=\prod_{i=1}^np(x_i, \theta_1, \theta_2, \cdots, \theta_k)
  \end{aligned}
  $$

记$L(\theta_1, \theta_2, \cdots, \theta_k)=\prod_{i=1}^np(x_i, \theta_1, \theta_2, \cdots, \theta_k)$为样本的似然函数

2. X是连续型总体，其概率密度为$f(x, \theta_1, \theta_2, \cdots, \theta_k)$则称$L(\theta_1, \theta_2, \cdots, \theta_k)=\prod_{i=1}^nf(x_i, \theta_1, \theta_2, \cdots, \theta_k)$为其样本的似然函数。
似然函数值的大小实质上反映的是该样本值出现的可能性的大小。

#### 极大似然估计的方法

对给定样本值$x_1, x_2, \cdots, x_n$，选取$\theta_1, \theta_2, \cdots, \theta_k$使其似然函数$L(\theta_1, \theta_2, \cdots, \theta_k)$达到最大值，即求$\hat{\theta_i}=\theta_i(x_1, x_2, \cdots, x_n),i=1, 2, \cdots, k$使得$L(\hat{\theta_1}, \hat{\theta_2}, \cdots, \hat{\theta_k})=max L(\theta_1, \theta_2, \cdots, \theta_k)$

这样得到的估计值称为未知参数$\theta_1, \cdots, \theta_k$的极大似然估计值，对应的统计量称为极大似然估计量。

步骤：

1. 由总体分布和所给样本求得似然函数$$L(\theta_1, \theta_2, \cdots, \theta_k)=\prod_{i=1}^nf(x_i, \theta_1, \theta_2, \cdots, \theta_k)$$
2. 求似然函数$L(\theta_1, \theta_2, \cdots, \theta_k)$的对数函数$$ln L(\theta_1, \theta_2, \cdots, \theta_k)=ln\prod_{i=1}^nf(x_i, \theta_1, \theta_2, \cdots, \theta_k)$$
3. 解方程组$$\begin{cases}\frac{\partial lnL(\theta_1, \theta_2, \cdots, \theta_k)}{\partial \theta_1}=0\\\frac{\partial lnL(\theta_1, \theta_2, \cdots, \theta_k)}{\partial \theta_2}=0\\ \cdots \\ \frac{\partial lnL(\theta_1, \theta_2, \cdots, \theta_k)}{\partial \theta_k}=0 \end{cases}$$
4. 得未知参数$\theta_1, \cdots, \theta_k$的极大似然估计值$$\begin{cases}\hat{\theta_1}=\hat{\theta_1}(x_1, x_2, \cdots, x_n) \\ \cdots \\ \hat{\theta_k}=\hat{\theta_k}(x_1, x_2, \cdots, x_n) \end{cases}$$

说明：

1. 可证明极大似然估计具有下述性质

设$\theta$的函数$g=g(\theta)$是$\Theta$的实值函数，且有唯一反函数，若$\hat{\theta}$是$\theta$的极大似然估计，则$g(\hat{\theta})$也是$g(\theta)$的极大似然估计。此性质称为极大似然估计的不变性。

2. 当似然函数不是可微函数时，需用极大似然原理来求待估参数的极大似然估计。

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

## 假设检验的基本概念

假设检验是根据样本提供的信息推断假设是否合理，并作出接受或拒绝提出的假设的决定。

### 假设检验的种类

+ 参数假设检验：总体分布已知，对未知参数提出的假设进行检验
+ 非参数假设检验：总体分布未知，对总体分布形式或类型的假设进行检验。

在假设检验问题中，把要检验的假设称为原假设记为$H_0$，把原假设的对立面称为备择假设或对立假设，记为$H_1$。原假设$H_0$和备择假设$H_1$两者中必有且仅有一个为真。

### 显著性检验的推理方法和基本步骤

1. 假设检验的理论依据：实际推断原理（小概率原理）。小概率事件在一次试验中几乎是不可能发生的
2. 假设检验是概率意义下的反证法
3. 不否定$H_0$并不是肯定$H_0$一定对，而是说差异不够显著，没达到足以否定$H_0$的程度。

假设检验的一般步骤

1. 根据实际问题的要求，充分考虑和利用已知的背景知识，提出原假设$H_0$和备择假设$H_1$
2. 给定显著性水平$\alpha$，选取检验统计量，并确定其分布
3. 由$P\{拒绝H_0|H_0为真\}=\alpha$确定$H_0$的拒绝域的形式
4. 由样本值求得检验统计量的观察值，若观察值在拒绝域内则拒绝原假设，否则接受。

### 两类错误

+ 第一类错误（弃真错误）：原假设$H_0$为真但拒绝了原假设
+ 第二类错误（取伪错误）：原假设$H_0$不为真但接受了原假设

记$P\{拒绝H_0|H_0为真\}=\alpha$，$P\{接受H_0|H_0不真\}=\beta$

显然，显著性水平$\alpha$为犯第一类错误的概率。

处理原则：控制犯第一类错误的概率$\alpha$，然后若有必要，通过增大样本容量的方法减少犯第二类错误的概率$\beta$。

注：关于原假设与备择假设的选取

原假设和备择假设地位应当平等。但控制犯一类错误的概率$\alpha$的原则下使得采取拒绝$H_0$的决策变得较为慎重，即$H_0$得到特别的保护。因而通常把有把握的有经验的结论作为原假设，或者尽可能使后果严重的错误成为第一类错误。

## 正态总体的假设检验

### 单一正态总体均值$\mu$的假设检验

设总体$X\sim N(\mu, \sigma^2)$，$X_1, X_2, \cdots, X_n$是取自$X$的样本，样本均值$\overline{X}$, 样本方差$S^2$

1. 已知$\sigma^2=\sigma_0^2$时，总体均值$\mu$的假设检验

+ $\mu$的双边检验
原假设$H_0:\mu=\mu_0$,备择假设$H_1:\mu\neq \mu_0$
取检验统计量$U=\frac{\overline{X}-\mu_0}{\sigma_0 / \sqrt{n}}$
则拒绝域为：$W=\{|U|\ge z_{\alpha/2}\}$

+ $\mu$的单边检验
a. 原假设$H_0:\mu \le \mu_0$，备择假设$H_1:\mu > \mu_0$
检验统计量$U=\frac{\overline{X}-\mu_0}{\sigma_0 / \sqrt{n}}$
拒绝域为：$W=\{U \ge z_{\alpha}\}$
b. 原假设$H_0:\mu \ge \mu_0$，备择假设$H_1:\mu < \mu_0$
检验统计量$U=\frac{\overline{X}-\mu_0}{\sigma_0 / \sqrt{n}}$
拒绝域为：$W=\{U \le -z_{\alpha}\}$

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411210217.png)

2. $\sigma^2$未知时，总体均值$\mu$的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411210629.png)

### 单一正态总体方差$\sigma^2$的假设检验

+ 已知$\mu=\mu_0$时，总体方差$\sigma^2$的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411211031.png)

+ $\mu$未知时，总体方差$\sigma^2$的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411211220.png)

### 两个正态总体均值的假设检验

$X_1, X_2, \cdots, X_{n_1}$为取自总体$N(\mu_1, \sigma_1^2)$的样本

$Y_1, Y_2, \cdots, Y_{n_2}$为取自总体$N(\mu_2, \sigma_2^2)$的样本

且两总体相互独立

$\overline{X}, S_1^2$；$\overline{Y}, S_2^2$分别表示两样本的样本均值与样本方差

+ 已知$\sigma_1^2,\sigma_2^2$时，总体均值的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411211701.png)

+ $\sigma_1^2,\sigma_2^2$未知，但$\sigma_1^2=\sigma_2^2$时，总体均值的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411211802.png)

### 两个正态总体方差的假设检验

+ 已知$\mu_1, \mu_2$时，总体方差的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411212334.png)

+ $\mu_1,\mu_2$未知时，总体方差的假设检验

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411212419.png)

### （0-1）总体参数p的大样本检验

已知总体X服从（0-1）分布，其分布律为$$f(x;p)=p^x(1-p)^{1-x}, x=0,1$$则$E(X)=p, D(X)=p(1-p)$

现抽取容量为$n\quad (n>30)$的样本$X_1, X_2, \cdots, X_n$，样本均值为$\overline{X}$

+ 对参数p的双边检验
原假设$H_0:p=p_0$，备择假设$H_1:p\neq p_0$
当原假设$H_0:p=p_0$为真时，由独立同分布中心极限定理可知$$U=\frac{\overline{X}-p_0}{\sqrt{\frac{p(1-p)}{n}}}\stackrel{近似}{\sim}N(0,1)$$
因为$\overline{X}$是p的达到方差界的无偏估计，所以$U$的值应较集中在零附近，而$H_0:p=p_0$的拒绝域应体现为$|U|$偏大，即拒绝域应形如$W=\{|U|\ge K\}$
设显著性水平为$\alpha$，由$$U=\frac{\overline{X}-p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}\stackrel{近似}{\sim}N(0,1)$$得：$K=z_{\alpha/2}, W=\{|U| \ge z_{\alpha/2}\}$

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210411213339.png)

## p值检验法

一般咋一个假设检验中，利用观测值能够做出的拒绝原假设的最小显著性水平称为该检验的p值。按p值的定义，对于任意指定的显著性水平a，有以下结论：

1. 若$\alpha < p$值，则在显著性水平$α$下接收$H_0$
2. 若$\alpha \ge p$值，则在显著性水平$\alpha$下拒绝$H_0$

有了这两条结论就能方便地确定$H_0$的拒绝域，这种利用p值检验假设的方法称为p值检验法。

p值反映了样本信息中所包含的反对原假设$H_0$的依据的强度，p值是已经观测到的一个小概率事件的概率。p值越小，$H_0$越有可能不成立，说明样本信息中反对$H_0$的依据的强度越强、越充分。

一般若$p \le 0.01$称拒绝$H_0$的依据很强或称检验是高度显著的；若$0.01 < p \le 0.05$，称拒绝$H_0$的依据是强的或称检验是显著的；若$0.05 < p \le 0.1$，称拒绝$H_0$的依据是弱的或称检验是不显著的。若$p > 0.1$一般来说没有理由拒绝$H_0$

### p值的计算

用$X$表示检验用的统计量，样本数据算出的统计量的值记为$C$，当$H_0$为真时可以算出p值

左侧检验：$p=P\{X < C\}$

右侧检验：$p=P\{X > C\}$

双侧检验：$X$落在以$C$为端点的尾部区域概率的两倍
$$
p=
\begin{cases}
2P\{X>C\}, C在分布的右侧\\
2P\{X<C\}, C在分布的左侧
\end{cases}
$$

$$
(如果分布对称)=P\{|X|>|C|\}
$$

## 单因素方差分析

### 基本概念

将要考察的对象的某种特征称为指标，影响指标的各种因素称为因子，一般将因子控制在几个不同的状态上，每个状态称为因子的一个水平

若一项试验中只有一个因子在改变而其他保持不变则称为**单因素试验**，多个因子在改变则称为**多因素试验**。

### 单因素方差分析的数学模型

假设前提：设在单因素试验中，影响指标的因子A由s个水平$A_1, A_2, \cdots, A_s$，将每个水平$A_j$下要考察的指标作为一个总体称为部分总体，仍记为$A_j$，则共有s个总体，假设： 

1. 每个部分总体都服从正态分布，即$$A_j \sim N(\mu_j, \sigma_j^2), j=1,2,\cdots, s$$
2. 部分总体的方差都相等，即$$\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_s^2 = \sigma^2$$
3. 不同的部分总体下样本是互相独立的

其中$\mu_1, \mu_2, \cdots, \mu_s$和$\sigma^2$都是未知参数。

在水平$A_j$下进行$n_j$次独立试验，得样本$X_{1j}, X_{2j}, \cdots, X_{n_jj}$,则$X_{ij} \sim N(\mu_j, \sigma^2)$

记$\epsilon_{ij} = X_{ij} - \mu_j$，称其为随机误差，则$\epsilon_{ij} \sim N(0, \sigma^2)$，由此得单因素方差分析的数学模型：

$$
\begin{cases}
x_{ij}=\mu_j+\epsilon_{ij} &i=1,2,\cdots, n_j \\
\epsilon_{ij} \sim N(0, \sigma^2) &j=1,2,\cdots, s
\end{cases}
$$


各随机误差$\epsilon_{ij}$相互独立，$\mu_1, \mu_2, \cdots, \mu_s$和$\sigma^2$未知。

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210417153624.png)

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210417153646.png)

#### 单因素方差分析的任务：

根据样本提供的信息：

1. 检验假设：$H_0: \mu_1=\mu_2=\cdots=\mu_s, H_1:\mu_1, \mu_2, \cdots, \mu_s不全相等$
2. 求出未知参数$\mu_1, \mu_2, \cdots, \mu_s$和$\sigma^2$的估计量

### 单因素方差分析的假设检验

#### 偏差平方和及其分解

+ 总平方和：$$S_T=\sum_{j=1}^s\sum_{i=1}^{n_j}(X_{ij}-\overline{X})^2$$
+ 效应（组间平方和）：$$\begin{aligned}S_A& =\sum_{j=1}^s\sum_{i=1}^{n_j}(X_{\cdot j}-\overline{X})^2\\ &=\sum_{j=1}^sn_j(\overline{X}_{\cdot j}-\overline{X})^2\end{aligned}$$

说明：$S_A$反映了每个水平下的样本均值与样本总均值的差异，它是由因子A取不同水平引起的，所以称$S_A$是因子A的效应(组间)平方和

+ 误差（组内平方和）：$$S_E=\sum_{j=1}^s\sum_{i=1}^{n_j}(X_{ij}-\overline{X}_{\cdot j})^2$$

说明：$S_E$表示在每个水平下的样本值与该水平下的样本均值的差异，它是由随机误差引起的，因此称其为误差平方和

平方和分解公式$$S_T=S_A+S_E$$

总平方和=效应（组间）平方和+误差（组内）平方和

#### $S_A$和$S_E$的统计特征

定理：在单因素方差分析的模型下：

1. $\frac{S_E}{\sigma^2} \sim \mathcal{X}^2(n-s)$
2. $S_A$和$S_E$相互独立
3. $H_0:\mu_1=\mu_2=\cdots=\mu_s$为真时，$\frac{S_A}{\sigma^2}\sim \mathcal{X}^2(s-1), \frac{S_T}{\sigma^2}\sim \mathcal{X}^2(n-1)$

由定理1，有$$E(\frac{S_E}{\sigma^2})=n-s$$ $$E[\frac{S_E}{n-s}]=\sigma^2$$，即$\hat{\sigma}^2=\frac{S_E}{n-s}$是$\sigma^2$的无偏估计

结合定理1，2，3，有$$F=\frac{S_A/(s-1)}{S_E/(n-s)}\sim F(s-1, n-s)$$

![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210417155029.png)

单因素方差分析的假设检验：

1. 提出统计假设$H_0: \mu_1=\mu_2=\cdots=\mu_s, H_1:\mu_1, \mu_2, \cdots, \mu_s不全相等$
2. 取假设统计量$F=\frac{S_A/(s-1)}{S_E/(n-s)}$
3. 拒绝域：$W=\{F\ge F_\alpha(s-1,n-s)\}$

说明：如果组间差异比组内差异大得多，则说明各水平间有显著差异，$H_0$不真

单因素方差分析假设检验的步骤：

1. 提出统计假设$H_0: \mu_1=\mu_2=\cdots=\mu_s, H_1:\mu_1, \mu_2, \cdots, \mu_s不全相等$
2. 编制单因素试验数据表
3. 根据数据表计算$T_{\cdot \cdot}, \sum_{j=1}^s\sum_{i=1}^{n_j}x_{ij}^2, S_T, S_A, S_E$
4. 填制单因素方差分析表
![](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/20210417155803.png)
5. 检验，若$\frac{S_A/(s-1)}{S_E/(n-s)}\ge F_\alpha (s-1, n-s)$,则拒绝$H_0$

### 部分总体均值$\mu_j$和方差$\sigma^2$的估计

之前已经说明：

1. $\hat{\sigma}^2=\frac{S_E}{n-s}$是$\sigma^2$的无偏估计

又 $E(\overline{X}_{\cdot j}=\frac{1}{n_j}\sum_{i=1}^{n_j}E(X_{ij})=\mu_j$，所以

2. $\overline{X}_{\cdot j}$是$\mu_j$的无偏估计，$j=1,2,\cdots, s$

可以证明，对每个j，$\overline{X}_{\cdot j}$也是$\mu_j$的最小二乘估计

## 双因素方差分析

假定条件：

1. 每个总体都服从正态分布
2. 各个总体的方差必须相同
3. 观察值是独立的

### 双因素方差分析的数据结构

![image-20210417164947190](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417164947190.png)

+ $\overline{X}_{i\cdot}$是因素A的第i个水平下的各观察值的平均值$\overline{X}_{i\cdot}=\frac{\sum_{j=1}^bx_{ij}}{b} (i=1,2,\cdots, a)$
+ $\overline{X}_{\cdot j}$是因素B的第j个水平下的各观察值的平均值$\overline{X}_{\cdot j}=\frac{\sum_{i=1}^ax_{ij}}{a} (j=1,2,\cdots, b)$
+ $\overline{\overline{X}}$是全部kr个样本数据的总平均值$\overline{\overline{X}}=\frac{\sum_{i=1}^a\sum_{j=1}^bx_{ij}}{ab}$

提出假设：

1. 因素A：$H_0: \mu_1=\mu_2=\cdots=\mu_i=\cdots=\mu_a(\mu_i为第i个水平的均值), H_1:\mu_i(i=1,2,\cdots, a)不全相等$
2. 因素B：$H_0: \mu_1=\mu_2=\cdots=\mu_j=\cdots=\mu_b(\mu_j为第j个水平的均值), H_1:\mu_j(j=1,2,\cdots, b)不全相等$

构造检验的统计量

1. 检验$H_0$是否成立，需要确定检验的统计量
2. 构造统计量需要计算
    + 总离差平方和
    + 水平项平方和
    + 误差项平方和
    + 均方

计算总离差平方和SST

+ $\overline{\overline{x}}$全部观察值$x_{ij} (i=1,2,\cdots, a;j=1,2,\cdots, b)$，与总平均值的离差平方和

+ 反映全部观察值的离散情况

+ 计算公式为
  $$
  SST=\sum_{i=1}^a\sum_{j=1}^b(x_{ij}-\overline{\overline{x}})^2
  $$

计算SSA、SSB和SSE和各个平方和的关系

![image-20210417165105805](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417165105805.png)

计算均方MS

1. 各离差平方和的大小与观察值的多少有关，为消除观察值多少对离差平方和大小的影响，需要将其平均，这就是均方，也称为方差
2. 计算方法是用离差平方和除以相应的自由度
3. 三个平方和的自由度分别是
   + 总离差平方和*SST*的自由度为 *ab*-1
   + 因素*A*的离差平方和*SSA*的自由度为 *a*-1
   + 因素*B*的离差平方和*SSB*的自由度为 *b*-1
   + 随机误差平方和*SSE*的自由度为 $(a-1)\times(b-1) $

4. 因素A的均方，记为MSA，计算公式为
   $$
   MSA=\frac{SSA}{a-1}
   $$

5. 因素B的均方，记为MSB，计算公式为

$$
MSB=\frac{SSB}{b-1}
$$

6. 随机误差项的均方，记为MSE，计算公式为

$$
MSE=\frac{SSE}{(a-1)(b-1)}
$$

检验计算的统计量F

1. 为检验因素A的影响是否显著，采用下面的统计量

$$
F_A=\frac{MSA}{MSE}\sim F(a-1, (a-1)(b-1))
$$

2. 为检验因素B的影响是否显著，采用下面的统计量

$$
F_B=\frac{MSB}{MSE}\sim F(b-1, (a-1)(b-1))
$$

统计决策

将统计量的值$F$与给定的显著性水平$\alpha$的临界值$F_\alpha$进行比较，作出接受或拒绝原假设$H_0$的决策

1. 根据给定的显著性水平$\alpha$在$F$分布表中查找相应的临界值$F_\alpha$
2. 若$F_A\ge F_\alpha$，则拒绝原假设$H_0$，表明均值之间的差异是显著的，即所检验的因素A对观察值有显著影响
3. 若$F_B \ge F_\alpha$，则拒绝原假设$H_0$，表明均值之间有显著差异，即所检验的因素B对观察值有显著影响

双因素方差分析表

![image-20210417165131919](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417165131919.png)

## 有交互作用的双因素试验的方差分析

双因素有重复（有交互作用）试验资料表

![image-20210417165734194](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417165734194.png)

![image-20210417165808744](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417165808744.png)

![image-20210417165839314](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417165839314.png)

处理方法：把交互作用当成一个新因素处理，即每种搭配$A_iB_j$看作一个总体$X_{ij}$

基本假设：

1. $X_{ij}$相互独立
2. $X_{ij} \sim N(\mu_{ij}, \sigma^2)$，（方差齐性）

记$\mu=\frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b\mu_{ij}$，所有期望值的总平均$X_{ijk}-\mu_{ij} \sim N(0, \sigma^2)$反映随机误差记为$\epsilon_{ijk}$，由$X_{ijk}-\overline{X_{ij\cdot}}$反映，$\alpha_i=\frac{1}{b}\sum_{j=1}^b\mu_ij-\mu=\mu_{i\cdot}-\mu$的无偏估计为$\frac{1}{b}\sum_{j=1}^bX_{ij\cdot}-\frac{1}{abn}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^nX_{ijk}=\overline{X_{i \cdot \cdot}}-\overline{X}$反映$A_i$的效应，

$\beta_j=\frac{1}{a}\sum_{i=1}^a\mu_{ij}-\mu=\mu_{\cdot i}-\mu$的无偏估计为$\frac{1}{a}\sum_{i=1}^aX_{ij\cdot}-\frac{1}{abn}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^nX_{ijk}=\overline{X_{\cdot j \cdot}} -\overline{X}$反映$B_j$的效应$(\alpha \beta)_{ij}=\mu_{ij} - \mu - \alpha_i - \beta_j$的无偏估计为$\overline{X_{ij\cdot}} - \overline{X_{i\cdot}} - \overline{X_{\cdot j}} + \overline{X}$反映交互效应

于是$\mu_{ij} = \mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}，$$X_{ijk}=\mu_{ij}+\epsilon_{ijk}=\mu+\alpha_i+\beta_j+(\alpha \beta)_{ij}+\epsilon_{ijk}$

![image-20210417170855987](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417170855987.png)

![image-20210417170909530](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417170909530.png)

![image-20210417170929602](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417170929602.png)

![image-20210417170945210](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417170945210.png)

![image-20210417171009980](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210417171009980.png)

## 正交试验设计

### 正交试验设计的概念和原理

#### 正交试验设计的基本概念

正交试验设计是利用正交表来安排与分析多因素试验的一种设计方法。它是由试验因素的全部水平组合中，挑选部分有代表性的水平组合进行试验的，通过对这部分试验结果的分析了解全面试验的情况，找出最优的水平组合。

正交试验设计的**基本特点**是：用部分实验来代替全部试验，通过对部分试验结果的分析，了解全面试验的情况。

对于3因素3水平试验，若不考虑交互作用，可利用正交表$L_9(3^4)$安排，试验方案仅包含9个水平组合，就能反映试验方案包含27个水平组合的全面试验的情况，找出最佳的生产条件。

#### 正交试验设计的基本原理

在试验安排中 ，每个因素在研究的范围内选几个水平，就好比在选优区内打上网格 ，如果网上的每个点都做试验，就是全面试验。如上例中，3个因素的选优区可以用一个立方体表示，3个因素各取3个水平，把立方体划分成27个格点，反映在图上就是立方体内的27个$\cdot$。若27个网格点都试验，就是全面试验，其试验方案如表所示。

![image-20210418144914037](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418144914037.png)

![image-20210418144936433](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418144936433.png)

正交设计就是从选优区全面试验点（水平组合）中挑选出有代表性的部分试验点（水平组合）来进行试验。图中标有试验号的九个$\cdot$就是利用正交表$L_9(3^4)$的27个试验点中挑选出来的9个试验点，即

1. $A_1B_1C_1$
2. $A_2B_1C_2$
3. $A_3B_1C_3$
4. $A_1B_2C_2$
5. $A_2B_2C_3$
6. $A_3B_2C_1$
7. $A_1B_3C_3$
8. $A_2B_3C_1$
9. $A_3B_3C_2$

上述选择保证了A因素的每个水平与B因素、C因素的各个水平在试验中各搭配一次。对于A、B、C3个因素来说，是在27个全面试验点中选择9个试验点，仅是全面试验的三分之一。

9个试验点在选优区中分布式均衡的，在立方体的每个平面上，都恰是3个试验点；在立方体的每条线上也恰有一个试验点。

### 正交表及其基本性质

#### 正交表

下图是一张正交表，记号为$L_8(2^7)$，其中$L$代表正交表，右下角数字8代表8行，用这张正交表安排试验包含8个处理（水平组合），括号内的底数2表示因素的水平数，括号内2的指数7表示有7列，用这张正交表最多可以安排7个2水平因素。

![image-20210418145901054](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418145901054.png)

#### 正交表的基本性质

##### 正交性

1. 任意列中各水平都出现，且出现的次数相等。
2. 任两列之间各种不同水平的所有可能组合都出现，且出现的次数相等。

##### 代表性

一方面：

1. 任一列的各水平都出现，使得部分试验中包括了所有因素的所有水平
2. 任两列的所有水平组合都出现，使任意两因素间的试验组合为全面试验

另一方面：由于正交表的正交性，正交试验的试验点必然均衡分布在全面试验点中，具有很强的代表性。因此部分实验寻找的最优条件与全面试验所找的最优条件应有一致的趋势。

### 正交试验设计的基本程序

包括试验方案设计和试验结果分析两部分。

试验方案设计

<img src="https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418150738892.png" alt="image-20210418150738892" style="zoom:50%;" />

试验结果分析：

![image-20210418150808281](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418150808281.png)

### 正交试验的结果分析

#### 直观分析法——极差分析法

计算简便直观，简单易懂，是正交试验结果分析最常用的方法

![image-20210418151102800](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210418151102800.png)

$K_{jm}$是第j列因素m水平所对应的试验指标和，$\overline{K_{jm}}$为$K_{jm}$均值。有$K_{jm}$大小可以判断第j列因素优水平和优组合。

$R_j$是第j列元素的极差，反映了第j列因素水平波动时，试验指标的变动幅度。$R_j$越大，说明该因素对试验指标的影响越大。根据$R_j$大小，可以判断因素的主次顺序。

#### 不考察交互作用的试验结果分析

1. 确定实验因素的有水平和最优水平组合

分析A因素各水平对试验指标的影响。$A_1$的影响反应在1、2、3号试验中，$A_2$的影响反应在第4、5、6号实验中，$A_3$的影响反映在第7、8、9号试验中。

A因素的1水平所对应的试验指标之和为
$$
K_{A1}=y1+y2+y3 \quad k_{A1}=K_{A1} / 3 \\
K_{A2}=y4+y5+y6 \quad k_{A2}=K_{A2} / 3 \\
K_{A3}=y7+y8+y9 \quad k_{A3}=K_{A3} / 3
$$

2. 确定因素的主次顺序

根据极差$R_j$的大小，可以判断各因素对试验指标的影响主次。R越大表示该因素的水平变化对试验指标的影响越大。
$$
R=max(\overline{K_i})-min(\overline{K_i})
$$


3. 绘制因素与指标趋势图。



以上即正交试验极差分析的基本程序和方法。

#### 有交互作用的正交设计与结果分析

除表头设计和结果分析与之前的介绍略有不同外，其余基本相同。

### 正交试验结果的方差分析

方差分析思想、步骤同前。基本思想是将数据的总变异分解成因素引起的变异和误差引起的编译两部分，构造$F$统计量，作$F$检验，即可判断因素作用是否显著。

1. 偏差平方和分解

$$
SS_T=SS_{因素}+SS_{空列（误差）}
$$

2. 自由度分解

$$
df_T=df_{因素}+df_{空列（误列）}
$$

3. 方差

$$
MS_{因素}=\frac{SS_{因素}}{df_{因素}}， MS_{误差}=\frac{SS_{误差}}{df_{误差}}
$$

4. 构造$F$统计量

$$
F_{因素}=\frac{MS_{因素}}{MS_{误差}}
$$

5. 列方差分析表，作$F$检验。



## 一元线性回归

回归函数：当可控变量$X$和随机变量$Y$之间存在回归关系时，$Y$的数学期望$E(Y)$是可控变量$X$的取值$x$的函数，记为$\mu(x)$，即$E(Y)=\mu(x)$，称$\mu(x)$为回归函数。

一元线性回归数学模型的两个前提：

1. 线性相关假设：设$\mu(x)=a+bx$，这里a、b是与可控变量X的取值x无关的未知参数
2. 随机变量Y服从正态分布，$Y\sim N(\mu(x), \sigma^2)$，这里$\sigma^2$是与x无关的未知参数。

设随机变量$\epsilon=Y-\mu(x)$，称其为随机误差，则$\epsilon\sim N(0, \sigma^2)$

线性回归的任务：根据实验的观测值求出$a,b,\sigma^2$的估计量：$\hat{a}, \hat{b}, \hat{\sigma^2}$，从而对任何x得回归函数的估计量：$\hat{y}=\hat{a}+\hat{b}x$（称其为随机变量Y依变量X的线性回归方程，其直线称为回归直线）,从而为进一步的预测和控制提供依据。

### 未知参数$a$，$b$和$\sigma^2$的点估计

#### 最小二乘法

根据偏差的平方和为最小来选择待估参数的方法称为最小二乘法。由此方法得到的估计量称为最小二乘估计。

即对样本观察值：$(x_1, y_1), (x_2, y_2), \cdots, (x_n, y_n)$，求使得二元函数$Q(a,b)=\sum_{i=1}^n\epsilon_i^2=\sum_{i=1}^n(y_i-a-bx_i)^2$最小的a、b估计量

令
$$
\begin{cases}
\frac{\partial Q}{\partial a}=-2\sum_{i=1}^n(y_i-a-bx_i)=0 \\
\frac{\partial Q}{\partial b}=-2\sum_{i=1}^n(y_i-a-bx_i)x_i=0
\end{cases}
$$
即
$$
\begin{cases}
na+(\sum_{i=1}^nx_i)b=\sum_{i=1}^ny_i\\
(\sum_{i=1}^nx_i)a+(\sum_{i=1}^nx_i^2)b=\sum_{i=1}^nx_iy_i
\end{cases}
$$
用克莱姆法则借这个二元线性方程组得a、b的最小二乘估计：$\hat{b}=\frac{L_{xy}}{L_{xx}}$，$\hat{a}=\overline{y}-\hat{b}\overline{x}$

从而得到回归方程$\hat{y}=\hat{a}+\hat{b}x$

可以证明：

1. $\hat{a}$，$\hat{b}$也是a、b的极大似然估计
2. $\hat{a}$，$\hat{b}$是$y_1, y_2, \cdots, y_n$的线性函数
3. $\hat{a}$，$\hat{b}$是a,b的无偏估计，且是a,b的一切线性无偏估计量中方差最小的估计量。

称$\hat{y_i}=\hat{a}+\hat{b}x_i$为观察值$y_i$的回归值$(i=1,2,\cdots, n)$

由a、b的最小二乘估计的推导过程可得：

回归直线的两个重要特征：

1. $y_i$偏离回归值$\hat{y_i}$的总和为零，即

$$
\sum_{i=1}^n(y_i-\hat{y_i})=0
$$

2. 平面上n个点$(x_1, y_1), (x_2, y_2), \cdots, (x_n, y_n)$的几何中心$(\overline{x}, \overline{y})$落在回归直线上，即

$$
\overline{y}=\hat{a}+\hat{b}\overline{x}
$$



未知参数$\sigma^2$的估计

首先给出残差平方和（剩余平方和）和回归平方和的概念

残差平方和：$Q=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n(y_i-\hat{a}-\hat{b}x_i)^2$

回归平方和：$U=\sum_{i=1}^n(\hat{y_i}-\overline{y})^2$

可以证明：$\frac{Q}{\sigma^2}\sim \mathcal{X}^2(n-2)$

于是$E(\frac{Q}{\sigma^2})=n-2$，即$E(\frac{Q}{n-2})=\sigma^2$

所以$\hat{\sigma}^2=\frac{Q}{n-2}$是$\sigma^2$的无偏估计



残差平方和Q的计算方法

定理（平方和分解公式）：对容量为n的样本$(x_1, y_1), (x_2, y_2), \cdots, (x_n, y_n)$总成立
$$
\sum_{i=1}^n(y_i-\overline{y})^2=\sum_{i=1}^n(y_i-\hat{y}_i)^2+\sum_{i=1}^n(\hat{y}_i-\overline{y})^2
$$
即：$L_{yy}=Q+U$

