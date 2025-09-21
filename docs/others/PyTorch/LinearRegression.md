# 线性回归
机器学习算法之一，用于预测一个连续值。**回归分析方法**，通过拟合线性函数预测输出。通过继承nn.Module类实现

$$
y=\Sigma_{i=1}^{n}  \omega_i x_i+ \text{bias}
$$

- y: 预测值（目标值）
- x: 输入特征
- w： 参数（待学习权重）
- bias：偏置项

??? example
    ```py
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn as nn

    torch.manual_seed(42)

    X = torch.randn(100, 2)
    true_w = torch.tensor([2.0, 3.0])
    true_b = 4.0
    Y = X @ true_w + true_b + torch.randn(100) * 0.1 #加入噪声
    print(X[:5])
    print(Y[:5])

    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x) #返回预测结果
        

    model = LinearRegressionModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        model.train() #设置训练模式

        predictions = model(X)
        loss = criterion(predictions.squeeze(), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    print(f'Predicted weight: {model.linear.weight.data.numpy()}')
    print(f'Predicted bias: {model.linear.bias.data.numpy()}')

    with torch.no_grad():  # 评估时不需要计算梯度
        predictions = model(X)

    # 可视化预测与实际值
    plt.scatter(X[:, 0], Y, color='blue', label='True values')
    plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
    plt.legend()
    plt.savefig('output.png')  # 保存图像到文件
    ```