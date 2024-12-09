# coding: utf-8
import pickle
import numpy as np
# from ch06.weight_init_activation_histogram import sigmoid
from common.functions import sigmoid, softmax
from dataset.mnist import load_mnist


# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # x_train: 60000*784, t_train: 60000*1, x_test: 10000*784, t_test: 10000*1
    # normalize=True表示将输入数据正规化为0.0~1.0的值，flatten=True表示展开输入图像，one_hot_label=False表示标签为1维数组
    # 预处理是对神经网络的输入数据进行某种变换，使其变为适合输入到神经网络中的形式。这里的预处理是将图像数据转换为0.0~1.0的实数，即正规化
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))