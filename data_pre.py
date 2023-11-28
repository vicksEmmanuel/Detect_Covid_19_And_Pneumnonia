import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# 定义类别标签
class_labels = {'normal': 0, 'pneumonia': 1, 'covid': 2}

# 初始化数据和标签列表
data = []
labels = []

# 大文件夹路径
main_folder_path = "F:/Teesside/Intelligent-DSS/cw2/archive"

# 遍历每个类别的子文件夹
for category in os.listdir(main_folder_path):
    category_path = os.path.join(main_folder_path, category)
    label = class_labels[category]

    # 初始化当前类别的数据和标签列表
    category_data = []
    category_labels = []

    # 遍历当前类别的图像
    for filename in os.listdir(category_path):
        filepath = os.path.join(category_path, filename)

        # 使用OpenCV读取图像
        image = cv2.imread(filepath)

        if image is not None:
            # make the size in the dataset all the same
            image = cv2.resize(image, (224, 224))#pixel can be modified(224 for VGG)

            # The mean values of the RGB channels (red, green, blue) across all images in the training set were subtracted from each pixel in the input image.
            # The mean values are [103.939, 116.779, 123.68]
            #mean_values = [103.939, 116.779, 123.68]
            #image -= mean_values
            #if VGG is used remove the hashtag in line 39, 40 above

            # RGB channel reordering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 添加噪音
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            # 随机旋转 add robustion
            angle = np.random.randint(-10, 10)  # 随机旋转角度在[-10, 10]之间
            rows, cols, _ = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows))

            # 归一化
            image = image / 255.0 #normalization for VGG

            image_flattened = image.flatten()

            # 将图像和标签添加到当前类别的列表
            category_data.append(image_flattened)
            category_labels.append(label)

    # 将当前类别的数据和标签添加到总的列表
    data.extend(category_data)
    labels.extend(category_labels)

# 转换为NumPy数组
data = np.array(data)
labels = np.array(labels)

#print("Data shape:", data.shape)
#print("Labels shape:", labels.shape)
#check shape


# 将数据和标签合并为一个数组，方便划分
combined_data = np.column_stack((data, labels))

# 根据类别标签划分数据
normal_data = combined_data[combined_data[:, -1] == class_labels['normal']]
pneumonia_data = combined_data[combined_data[:, -1] == class_labels['pneumonia']]
covid_data = combined_data[combined_data[:, -1] == class_labels['covid']]

# 数据划分为训练集和测试集
train_normal, test_normal = train_test_split(normal_data, test_size=0.2, random_state=42)
train_pneumonia, test_pneumonia = train_test_split(pneumonia_data, test_size=0.2, random_state=42)
train_covid, test_covid = train_test_split(covid_data, test_size=0.2, random_state=42)

# 合并划分后的数据
train_data = np.concatenate((train_normal, train_pneumonia, train_covid), axis=0)
test_data = np.concatenate((test_normal, test_pneumonia, test_covid), axis=0)

# 分离数据和标签
train_data, train_labels = train_data[:, :-1], train_data[:, -1].astype(int)
test_data, test_labels = test_data[:, :-1], test_data[:, -1].astype(int)