import numpy as np
import matplotlib.pyplot as plt  # 用于可视化评估结果
from ucimlrepo import fetch_ucirepo  # 导入 `fetch_ucirepo` 函数，从 UCI 数据库加载数据集
from sklearn.preprocessing import LabelEncoder  # 导入 LabelEncoder 用于标签编码
import shap  # 导入 SHAP 库

# Fetch dataset (Iris dataset)
iris = fetch_ucirepo(id=53)  # 加载 Iris 数据集，ID 53 对应的是 Iris 数据集

# Data (as numpy arrays)
X = iris.data.features  # 提取数据集的特征（features）
y = iris.data.targets  # 提取数据集的目标（标签）

# Metadata
print("Metadata:", iris.metadata)  # 输出数据集元数据（如特征描述等）

# Variable information
print("Variables:", iris.variables)  # 输出特征变量的信息

# Split dataset into training and testing
# Combine features and target into one dataset for splitting
data = np.column_stack((X, y))  # 将特征和标签合并为一个数据集，每行是一个样本

# Split into training and testing sets (70% training, 30% testing)
def train_test_split_custom(data, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    n_samples = len(data)
    test_size = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)# 返回一个随机排列的范围
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data[train_indices], data[test_indices]# 随机划分测试集和训练集的模块定义

#随机划分测试集和训练集的使用
train_data, test_data = train_test_split_custom(data, test_size=0.3, random_state=42)

# 进行标签编码（手动实现）
def label_encode(data):
    label_mapping = {label: idx for idx, label in enumerate(np.unique(data))} #创建了一个字典 label_mapping，将 data 中的每个唯一值映射到一个唯一的整数索引。
    return np.array([label_mapping[label] for label in data]) #将数据集中每个标签（label）转换为对应的数值索引
train_data[:, -1] = label_encode(train_data[:, -1])
test_data[:, -1] = label_encode(test_data[:, -1])

# 进行数据标准化（手动实现）
def standardize(X):
    X = np.array(X, dtype=np.float64)
    if len(X.shape) == 1:  # 如果输入是一维数组（只有一个特征），调整形状为二维
        X = X.reshape(-1, 1)

    # 处理缺失值，如果有缺失值（NaN或inf），则使用均值填充
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        nan_mean = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X) | np.isinf(X), nan_mean, X)

    # 计算均值和标准差
    means = np.mean(X, axis=0)# 均值
    stds = np.std(X, axis=0)

    stds = np.where(stds == 0, 1, stds)  # 如果标准差为0，设置为1，防止除以0

    # 对每个特征进行标准化
    return {'X': X, 'means': means, 'stds': stds}  # 返回一个字典

# 使用标准化处理训练数据
standardized_train_data = standardize(train_data[:, :-1])  # 使用训练集的特征进行标准化
X_train = standardized_train_data['X']
train_means = standardized_train_data['means']
train_stds = standardized_train_data['stds']

### CART回归树类
class CART_tree:
    def __init__(self):  # 构造函数，初始化相关参数
        self.min_leaf_size = 5  # 叶子节点的最小样本数设置为 5
        self.varThres = 0  # 分裂所需的最小方差减少阈值设置为 0
        self.max_depth = 10  # 树最大深度设为 10
        self.feature_importance = np.zeros(4)  # 假设数据集有4个特征，初始化特征重要性

    def datasetSplit(self, dataset, feaNum, thres):  # 根据特征和阈值划分数据集
        dataL = dataset[np.nonzero(dataset[:, feaNum] < thres)[0], :]  # 左子集（特征小于阈值）
        dataR = dataset[np.nonzero(dataset[:, feaNum] >= thres)[0], :]  # 右子集（特征大于等于阈值）
        return dataL, dataR  # 返回左右子集

    def getAllVar(self, dataset):  # 计算数据集的总方差（用于回归任务）
        return np.var(dataset[:, -1]) * len(dataset)  # 方差乘以样本数

    def findFeatureAndThresParallel(self, feature, dataset):  # 查找当前特征的最佳分割阈值
        m = len(dataset)  # 获取样本数

        # 使用 argsort 排序数据集，保留数据集的结构
        sorted_indices = np.argsort(dataset[:, feature])  # 获取按特征列排序后的索引
        dataset_t = dataset[sorted_indices]  # 使用这些索引重新排序数据集

        # 提取排序后的特征值和标签
        thresList = dataset_t[:, feature]  # 获取排序后的特征值
        sum_List = np.cumsum(dataset_t[:, -1])  # 计算标签的累计和
        sq_sum_List = np.cumsum(np.square(dataset_t[:, -1]))  # 计算标签的累计平方和

        sum = sum_List[-1]  # 获取总和
        sq_sum = sq_sum_List[-1]  # 获取平方和

        new_thresList, index = np.unique(thresList, return_index=True)  # 去重并获取唯一值的索引

        left_size = index  # 左子集的样本数
        right_size = m - left_size  # 右子集的样本数

        left_sum = sum_List[left_size - 1]  # 左子集的标签累计和
        left_sq_sum = sq_sum_List[left_size - 1]  # 左子集的标签平方和
        right_sum = sum - left_sum  # 右子集的标签累计和
        right_sq_sum = sq_sum - left_sq_sum  # 右子集的标签平方和

        left_size[0] = 1  # 防止除零错误

        var_left = left_sq_sum / left_size - np.square(left_sum / left_size)  # 计算左子集的方差
        var_right = right_sq_sum / right_size - np.square(right_sum / right_size)  # 计算右子集的方差
        total_lost = var_left * left_size + var_right * right_size  # 计算总损失（均方误差）

        if len(thresList) <= 2 * self.min_leaf_size:  # 如果候选分割点过少，则返回
            return

        l = index >= self.min_leaf_size  # 左子集样本数满足最小叶子节点条件
        r = index < m - self.min_leaf_size  # 右子集样本数满足最小叶子节点条件
        listRange = np.nonzero(l & r)[0]  # 找到有效的分割点

        if len(listRange) == 0:  # 如果没有有效的分割点，则返回
            return

        index = np.argmin(total_lost[listRange], axis=0)  # 找到最小损失对应的分割点
        if total_lost[listRange[0] + index] < self.varThres:  # 更新最优分割点
            self.varThres = total_lost[listRange[0] + index]
            self.bestFeature = feature
            self.bestThres = new_thresList[listRange[0] + index]

            # 更新特征重要性
            self.feature_importance[feature] += total_lost[listRange[0] + index]  # 重要性越大，值越大

    def chooseBestFeature(self, dataset, featureList, max_depth):  # 选择最佳特征和阈值
        if len(set(dataset[:, -1])) == 1 or max_depth == 0:  # 如果标签相同或达到最大深度，停止分裂
            return None, np.mean(dataset[:, -1])

        if len(featureList) == 1:  # 如果没有剩余特征，停止分裂
            return None, np.mean(dataset[:, -1])

        totalVar = self.getAllVar(dataset)  # 计算当前数据集的总方差
        self.bestFeature = -1  # 初始化最优特征
        self.bestThres = float('-inf')  # 初始化最优阈值
        self.varThres = np.inf  # 初始化最低误差

        for feature in featureList:  # 遍历所有特征，寻找最佳分割点
            self.findFeatureAndThresParallel(feature, dataset)

        if totalVar - self.varThres < self.varThres:  # 如果方差减少不足，停止分裂
            return None, np.mean(dataset[:, -1])
        # 在决策树算法中，节点的划分标准通常基于数据的纯度度量，如信息增益、基尼指数或方差等。在回归树的构建过程中，方差是衡量数据纯度的重要指标。
#如果当前节点的方差减少量不足以超过预设的阈值，则认为进一步划分不会显著提升模型性能，因此停止分裂，并将该节点的输出设为当前数据的均值。
        dataL, dataR = self.datasetSplit(dataset, self.bestFeature, self.bestThres)  # 分割数据集
        if len(dataL) < self.min_leaf_size or len(dataR) < self.min_leaf_size:  # 如果子集太小，停止分裂
            return None, np.mean(dataset[:, -1])

        return self.bestFeature, self.bestThres

    def createTree(self, random_dataset, max_depth=10):
        """
        递归构建CART回归树。
        :param random_dataset: 当前节点的子集数据
        :param max_depth: 最大树深度
        :return: 构建好的回归树（字典形式）
        """
        n = 5

        # 如果只剩下一个类别，直接返回类别的平均值
        if len(set(random_dataset[:, -1])) == 1 or max_depth == 0:
            return np.mean(random_dataset[:, -1])

        # 计算当前数据集的所有特征的方差
        featureList = list(range(n - 1))  # 特征列（去掉标签列）

        # 获取最佳分割特征和阈值
        bestFeat, bestThres = self.chooseBestFeature(random_dataset, featureList, max_depth)

        # 如果没有合适的分割点，返回叶子节点的均值
        if bestFeat is None:
            return np.mean(random_dataset[:, -1])

        # 根据最佳分割特征和阈值分割数据
        dataL, dataR = self.datasetSplit(random_dataset, bestFeat, bestThres)

        # 创建当前节点的字典，记录分割特征和分割阈值
        regTree = {
            'spliteIndex': bestFeat,
            'spliteValue': bestThres,
            'left': self.createTree(dataL, max_depth - 1),
            'right': self.createTree(dataR, max_depth - 1)
        }
        return regTree

    def isTree(self, tree):  # 判断是否为树（叶子节点）
# 叶子节点（Leaf Node）是指没有任何子节点的节点，即其度为零的节点。这些节点位于树的最底层，代表树的终端部分。
        return isinstance(tree, dict)

    def predict(self, tree, param):
        """
        使用训练好的树进行单个样本的预测。
        :param tree: 训练好的CART回归树
        :param param: 单个测试样本
        :return: 预测值
        """
        if not self.isTree(tree):  # 如果是叶子节点，返回预测值
            return float(tree)

        # 根据当前节点的分割特征，选择分裂方向
        if param[tree['spliteIndex']] < tree['spliteValue']:
            if not self.isTree(tree['left']):
                return float(tree['left'])
            else:
                return self.predict(tree['left'], param)  # 递归左子树
        else:
            if not self.isTree(tree['right']):
                return float(tree['right'])
            else:
                return self.predict(tree['right'], param)  # 递归右子树


# 随机森林实现
class RandomForest:
    def __init__(self, n):  # 构造函数，初始化随机森林
        self.treeNum = n  # 随机森林中树的数量
        self.treeList = []  # 存储所有树的列表
        self.ct = CART_tree()  # 实例化CART树类

    def fit(self, dataset, jobs=1):  # 训练随机森林
        m, n = len(dataset), len(dataset[0])
        for i in range(self.treeNum):  # 为每棵树训练随机数据集
            data_t = np.random.choice(range(m), m, replace=True)  # 随机采样数据
            random_dataset = dataset[data_t, :]  # 创建随机数据集
            self.treeList.append(self.ct.createTree(random_dataset))  # 训练树并添加到森林中

    def predict(self, testData):  # 预测测试数据
        result = []
        for i in range(len(testData)):  # 对每个样本进行预测
            res = []
            for tree in self.treeList:  # 对每棵树进行预测
                res.append(self.ct.predict(tree, testData[i]))  # 使用CART树进行预测
            result.append(res)  # 保存每棵树的预测结果

        # 对每个样本的预测值取平均
        result = np.mean(result, axis=1)
        return result

# 计算R平方损失（评估指标）
def R2Loss(y_test, y_true):
    y_test = np.array(y_test, dtype=np.float64)
    y_true = np.array(y_true, dtype=np.float64)
    return 1 - (np.sum(np.square(y_test - y_true))) / (np.sum(np.square(y_true - np.mean(y_true))))


# 计算均方误差（MSE）
def MSE(y_test, y_true):
    y_test = np.array(y_test, dtype=np.float64)
    y_true = np.array(y_true, dtype=np.float64)
    return np.mean(np.square(y_test - y_true))


# 计算均方根误差（RMSE）
def RMSE(y_test, y_true):
    return np.sqrt(MSE(y_test, y_true))


# 绘制评估指标的可视化图
def plot_metrics(r2_loss, mse, rmse):
    metrics = ['R² Loss', 'MSE', 'RMSE']
    values = [r2_loss, mse, rmse]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.show()


# 绘制特征重要性
def plot_feature_importance(importance, feature_names):
    importance = importance.flatten()  # 确保重要性是一个一维数组
    feature_names = list(feature_names)  # 确保特征名称是一个列表

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Based on Random Forest')
    plt.show()

# 使用 SHAP 的 KernelExplainer 来计算特征重要性
def calculate_shap_values(rf, X_train):
    # 定义预测函数，用于 SHAP 计算
    def predict_fn(X):
        return rf.predict(X)  # 使用训练好的随机森林进行预测

    # 使用 KernelExplainer 来计算 SHAP 值
    explainer = shap.KernelExplainer(predict_fn, X_train)  # SHAP的解释器
    shap_values = explainer.shap_values(X_train)  # 计算 SHAP 值

# KernelExplainer 是 SHAP 提供的一个模型无关（model-agnostic）的解释器，适用于任何模型。 ￼
# 它通过对特征进行扰动，观察模型输出的变化，来估计每个特征的 Shapley 值。 
  
    return shap_values, explainer

# 创建并训练随机森林模型
rf = RandomForest(10)  # 使用10棵树的随机森林
rf.fit(train_data, jobs=1)  # 训练随机森林

# 计算 SHAP 值
shap_values, explainer = calculate_shap_values(rf, X_train)

# 反标准化 SHAP 值
shap_values_original = (shap_values * train_stds) + train_means

# 获取特征名称
feature_names = iris.data.feature_names  # 获取 Iris 数据集的特征名称

# 绘制反标准化后的 SHAP 值图
shap.summary_plot(shap_values_original, X_train, feature_names=feature_names)

# 如果你想查看单个样本的 SHAP 贡献，可以使用以下代码：
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_train[0], feature_names=feature_names)
if __name__ == "__main__":  # 主程序入口
    print("Training the Random Forest")

    rf = RandomForest(10)  # 使用10棵树的随机森林
    rf.fit(train_data, jobs=1)  # 训练随机森林

    print("Predicting on the training dataset")
    result = rf.predict(train_data[:, :-1])  # 对训练集进行预测

    # 计算R2得分
    r2_loss = R2Loss(result, train_data[:, -1].flatten())
    print("R2 Loss on training data:", r2_loss)

    # 计算MSE和RMSE
    mse = MSE(result, train_data[:, -1].flatten())
    rmse = RMSE(result, train_data[:, -1].flatten())
    print("MSE on training data:", mse)
    print("RMSE on training data:", rmse)

    # 绘制评估指标的可视化图
    plot_metrics(r2_loss, mse, rmse)

