# 矩阵论

# $\lambda$矩阵

定义：设$a_{ij}(\lambda)(i=1,2,\cdots, m;j=1,2,\cdots, n)$为数域F上的多项式，称
$$
A= \left[
\begin{matrix} 
a_{11}(\lambda) \quad a_{12}(\lambda) \quad \cdots \quad a_{1n}(\lambda) \\
a_{21}(\lambda) \quad a_{22}(\lambda) \quad \cdots \quad a_{2n}(\lambda) \\
\cdots \quad \cdots \quad \cdots \quad \cdots \\
a_{m1}(\lambda) \quad a_{m2}(\lambda) \quad \cdots \quad a_{mn}(\lambda) \\
\end{matrix} 
\right]
$$
为多项式矩阵或$\lambda$矩阵



定义：若$\lambda$矩阵$A(\lambda)$有一个$r(r\ge 1)$阶子式不为0，而所有$r+1$阶子式（如果有的话）全为0，则称$A(\lambda)$的秩为r，记为
$$
rank A(\lambda)=r
$$
零矩阵的秩为0

定义：一个 *n*阶 $\lambda$矩阵称为可逆的，如果有一个*n* 阶 $\lambda$矩阵$B(\lambda)$，满足
$$
A(\lambda)B(\lambda) = I
$$


这里 $I$ 是 n 阶单位矩阵. $B(\lambda)$称为$A(\lambda)$矩阵的逆矩阵，记为$A^{-1}(\lambda)$

 TODO: P141-142



### $\lambda$矩阵的Smith标准型

#### 不变因子和初等因子

定理：任意一个非零的$m\times n$型的$\lambda$矩阵都等价于一个对角矩阵，即
$$
A= \left[
\begin{matrix} 
d_1(\lambda) & & & & & & \\
& d_2(\lambda) & & & & & \\
& & \cdots & & & & \\
& & & d_r(\lambda) & & & \\
& & & & 0 & & \\
& & & & & \cdots & \\
& & & & & & 0
\end{matrix} 
\right]
$$
其中$r\ge 1$，$d_i(\lambda)$是首项系数为1的多项式，$d_i(\lambda)|d_{i+1}(\lambda), i=1,2,\cdots, r-1$，称这种形式的$\lambda$矩阵为$A(\lambda)$的smith标准型，$d_i(\lambda)$为不变因子。

![image-20210529101431759](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529101431759.png)

定义所有幂指数不为0的因式$(\lambda-\lambda_j)^{k_ij}$为$A(\lambda)$的初等因子

注意对角线应当满足一次相除性

![image-20210529101611834](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529101611834.png)

#### 行列式因子

![image-20210529101701048](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529101701048.png)

k阶子式非零

![image-20210529102001638](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102001638.png)

$A(\lambda)$的Smith标准型是唯一的

![image-20210529102057804](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102057804.png)

![image-20210529102158652](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102158652.png)

仅仅初等因子组相同不能保证等价。

![image-20210529102240181](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102240181.png)

## 矩阵相似的条件

![image-20210529102258080](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102258080.png)

![image-20210529102336899](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102336899.png)

![image-20210529102428629](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102428629.png)

![image-20210529102555435](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102555435.png)

![image-20210529102611174](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102611174.png)

![image-20210529102622914](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102622914.png)



## Jordan标准型

![image-20210529102804220](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102804220.png)

![image-20210529102817176](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102817176.png)

![image-20210529102909677](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102909677.png)

![image-20210529102922526](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529102922526.png)

定理：A可以对角化的充分必要条件是，A的初等因子都是一次因式

![image-20210529103110028](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103110028.png)

![image-20210529103302899](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103302899.png)

![image-20210529103309137](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103309137.png)

![image-20210529103320931](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103320931.png)

### 相似变换矩阵

![image-20210529103537468](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103537468.png)

![image-20210529103550827](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103550827.png)

### Jordan标准型的幂

![image-20210529103850049](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103850049.png)

![image-20210529103953223](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529103953223.png)

![image-20210529104000331](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104000331.png)

![image-20210529104012069](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104012069.png)

## Hamilton-Cayley定理

![image-20210529104126443](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104126443.png)

### 零化多项式与最小多项式

![image-20210529104405399](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104405399.png)

![image-20210529104412357](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104412357.png)

#### 零化多项式与最小多项式的关系

![image-20210529104609240](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104609240.png)

![image-20210529104740590](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104740590.png)

## 矩阵的酉相似

### 内积空间

![image-20210529104816546](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529104816546.png)

![image-20210529105709207](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529105709207.png)

### 度量矩阵和Hermite矩阵

![image-20210529105803479](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529105803479.png)

![image-20210529105902534](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529105902534.png)

![image-20210529105910185](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529105910185.png)

![image-20210529110506659](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110506659.png)

### 内积空间的度量

![image-20210529110757018](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110757018.png)

#### Cauchy-Schwardz不等式

![image-20210529110822479](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110822479.png)

![image-20210529110846083](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110846083.png)

![image-20210529110948406](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110948406.png)

![image-20210529110958335](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210529110958335.png)

![image-20210530141238242](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530141238242.png)

![image-20210530141603345](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530141603345.png)

![image-20210530141641387](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530141641387.png)

### $C^m$空间中标准正交基

![image-20210530141842891](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530141842891.png)

![image-20210530141854439](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530141854439.png)

![image-20210530142009186](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530142009186.png)

#### Schmidt正交化

Schmidt正交化是把非正交基变为正交基

假设是n维空间，先让$x_1$和$x_2$正交，让$x_3$在这两个向量张成的空间作垂线从而使得$x_3$转换成与$x_1$和$x_2$分别正交的向量，以此类推。

![image-20210530143250832](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530143250832.png)

![image-20210530143315201](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530143315201.png)

![image-20210530143325230](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530143325230.png)

### 单位化

![image-20210530143410448](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530143410448.png)



## 酉空间分解与投影

#### 正交补子空间

![image-20210530144255945](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530144255945.png)

![image-20210530144446033](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530144446033.png)

#### 正交补空间的计算

![image-20210530145047792](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530145047792.png)

![image-20210530145057053](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530145057053.png)

![image-20210530145555090](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530145555090.png)

#### 投影

![image-20210530145704887](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530145704887.png)

#### 点到空间的距离与最小二乘法

![image-20210530145812291](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530145812291.png)

![image-20210530150354358](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530150354358.png)

![image-20210530150546299](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530150546299.png)

![image-20210530150641594](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530150641594.png)

![image-20210530150651052](https://cdn.jsdelivr.net/gh/lensherrggg/cloudimg@main/image-20210530150651052.png)

#### 正交投影

