# **数据处理与加载**
## **Dataset**
- **作用**：允许你自己的数据源中创建数据集。
- 需要继承并实现
    - `_len_(self)`：返回数据集样本数量
    - `_getitem_(self, idx)`：通过索引返回一个样本

??? example "自定义 Dataset"
    ```py
    import torch
    from torch.utils.data import Dataset

    # 自定义数据集类
    class MyDataset(Dataset):#继承
        def __init__(self, X_data, Y_data):
            self.X_data = X_data
            self.Y_data = Y_data

        def __len__(self):
            return len(self.X_data)

        def __getitem__(self, idx):
            x = torch.tensor(self.X_data[idx], dtype = torch.float32)

            y = torch.tensor(self.Y_data[idx], dtype = torch.float32)
            return x, y

    X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    Y_data = [1, 0, 1, 0]

    dataset = MyDataset(X_data, Y_data)
    ```

## **DataLoader**
**作用**：从 Dataset 按批次（batch）加载数据，允许批量读取数据、多线程加载，提高训练效率
- dataset
- batch_size: 每次加载的样本数量。
- shuffle: 是否对数据进行洗牌，通常训练时需要将数据打乱。
- drop_last: 如果数据集中的样本数不能被 batch_size 整除，设置为 True 时，丢弃最后一个不完整的 batch。
- num_workers：多线程加速

??? example "DataLoader"
    ```py
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    # 自定义数据集
    class MyDataset(Dataset):
        def __init__(self, data, labels):
            # 数据初始化
            self.data = data
            self.labels = labels

        def __len__(self):
            # 返回数据集大小
            return len(self.data)

        def __getitem__(self, idx):
            # 按索引返回数据和标签
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label

    # 生成示例数据
    data = torch.randn(100, 5)  # 100 个样本，每个样本有 5 个特征
    labels = torch.randint(0, 2, (100,))  # 100 个标签，取值为 0 或 1

    # 实例化数据集
    dataset = MyDataset(data, labels)
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

    # 遍历 DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"批次 {batch_idx + 1}")
        print("数据:", batch_data)
        print("标签:", batch_labels)
        if batch_idx == 2:  # 仅显示前 3 个批次
            break
    ```
## **transforms**
PyTorch 提供了 torchvision.transforms 模块来进行常见的图像预处理和增强操作，如旋转、裁剪、归一化等。`from torchvision import transforms`

- transforms.Compose()：将多个变换操作组合在一起。
- transforms.Resize()：调整图像大小。`transform = transforms.Resize((256, 256))`
- transforms.ToTensor()：将图像转换为 PyTorch 张量，值会被归一化到 [0, 1] 范围。
- transforms.Normalize(mean, std)：标准化图像数据，通常使用预训练模型时需要进行标准化处理，对图像进行标准化，使数据符合零均值和单位方差 `transform = transforms.Normalize(mean = [0.5], std = [0.5])`

$$
\text{output[channel] = (input[channel]-mean[channel])/std[channel]}
$$

- transforms.CenterCrop(size)：从图像中心裁剪指定大小的区域。
- RandomCrop: 从图像中是随机裁剪指定大小。 `transform = transforms.RandomCrop(128)`
- RandomHorizontalFlip: 一定概率水平翻转图像 `transform = transforms.RandomHorizontalFlip(p = 0.5)`
- RandomRotation: 随机旋转一定角度 `transform = transforms.RandomRotation(degrees = 30) #-30~+30`
- ColorJitter：随机改变图像亮度/对比度/饱和度/色调 `transform = transformers.ColorJitter(brightness=0.5, contrast=0.5)` 
     - brightness
     - contrast
     - saturation
     - hur
- Compose: 组合变换，按顺序应用

??? example "transforms"
    ```py
    import torchvision.transforms as transforms
    from PIL immport Image

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    image = Image.open('image.jpg')

    image_tensor = image.transform(image)
    ```

## **torchvision.datasets**
对于图像数据集，torchvision.datasets 提供了许多常见数据集（如 CIFAR-10、ImageNet、MNIST 等）以及用于加载图像数据的工具。

??? example "load FashionMNIST Dataset"
    ```py
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt


    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,#训练集/测试集
        download=True,
        transform=ToTensor()
    )
    ```

## **用多个数据源**
ConcatDataset 和 ChainDataset 把多个来自不同文件夹的数据合成一个数据集。

??? example "ConcatDataset"
    ```py
    from torch.utils.data import ConcatDataset

    combined_dataset = ConcatDataset([dataset1, dataset2])
    combined_loader = Dataloader(combined_dataset, batch_size=64, shuffle=True)
    ```