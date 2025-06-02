!!! info ""
    参考[B站视频](https://www.bilibili.com/video/BV1R4PRemES9?spm_id_from=333.788.videopod.episodes&vd_source=8729f4bfa3d735996f245c23d2f999dc&p=6)

## 1 线性回归
### 回归分析
- 定义：根据数据，确定两种或两种以上变量间相互依赖的定量关系。

- 公式表达：
  $$
  y=f(x_1,x_2...x_n)
  $$
### 线性回归
- 定义：回归分析中，变量与因变量存在线性关系
- 函数表达式：$$
y=ax+b
$$

### 回归问题求解
- 损失函数：$$
mimimize\{\frac{1}{2m}\sum_{i=1}^m(h_i-y_i)^2\}
$$
- 梯度下降法：寻找**极小值**。
$$
p_{i+1}=p_i-\alpha\frac{\partial}{\partial p_i}f(p_i)（\alpha:步长）
$$
??? note
    $$
    J=\frac{1}{2m}\sum_{i=1}^m(h_i-y_i)^2=\frac{1}{2m}\sum_{i=1}^m(ax_i+b-y_i)^=g(a,b)
    $$

    $$
    temp_a=a-\alpha\frac{\partial}{\partial a}g(a,b)=a-\alpha\frac{1}{m}\sum_{i=1}^m(ax_i+b-y_i)x_i
    $$

    $$
    temp_b=b-\alpha\frac{\partial}{\partial b}g(a,b)=b-\alpha\frac{1}{m}\sum_{i=1}^m(ax_i+b-y_i)
    $$

## 2 逻辑回归
### 单一变量
- 常用于解决分类任务
- 方程：
$$
Y=\frac{1}{1+e^{-x}}
$$
??? example "以x = 0 为分界线"
    $$
    Y=\frac{1}{1+e^{-x}}=\frac{1}{1+e^{0}}=\frac{1}{2}
    $$

    $$
    y=f(x)=\left\{\begin{array}{ll}0,&x<0\\1,&x\geq0\end{array}\right.
    $$
### 多变量
- 方程：
$$
Y=\frac{1}{1+e^{-g(x)}}
$$

$$
g(x)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_2^2
$$