import numpy as np
from matplotlib import pyplot as plt
import math


def procrustes_dis(obj1, obj2):
    temp = 0
    for i in range(obj1.shape[0]):
        temp += (obj1[i][0] - obj2[i][0]) ** 2 + (obj1[i][1] - obj2[i][1]) ** 2
    return math.sqrt(temp)


def remove_translate(obj):
    center = np.mean(obj, 0)
    obj -= center


def remove_scale(obj):
    n = obj.shape[0]
    s = 0
    for i in range(n):
        s += obj[i][0] ** 2 + obj[i][1] ** 2
    s = math.sqrt(s / n)
    obj /= s


def remove_translate_all(data):
    for i in range(data.shape[0]):
        remove_translate(data[i])


def remove_scale_all(data):
    for i in range(data.shape[0]):
        remove_scale(data[i])


def calc_angle(ref, obj):
    n = ref.shape[0]
    sum0 = 0.0
    sum1 = 0.0
    for i in range(n):
        sum0 += obj[i][0] * ref[i][1] - obj[i][1] * ref[i][0]
        sum1 += obj[i][0] * ref[i][0] + obj[i][1] * ref[i][1]
    return math.atan(sum0 / sum1)


def align(ref, obj):
    theta = calc_angle(ref, obj)
    cos = math.cos(theta)
    sin = math.sin(theta)
    for i in range(obj.shape[0]):
        x = obj[i][0]
        y = obj[i][1]
        obj[i][0], obj[i][1] = cos * x - sin * y, sin * x + cos * y


def align_all(ref, data):
    for i in range(data.shape[0]):
        align(ref, data[i])


def mean_shape(data):
    return np.mean(data, 0)


def read_data(filename):
    f = open(filename)

    ans = []

    for line in f:
        points = line.split(" ")
        ans.append(np.reshape([float(i) for i in points], (-1, 2)))

    f.close()

    return np.array(ans, dtype=np.float)


def plot(lines, limit):
    if limit <= 0:
        limit = lines.shape[0]
    for i in range(limit):
        plt.fill(lines[i, :, 0], lines[i, :, 1], facecolor="none", edgecolor="purple")
        plt.scatter(lines[i, :, 0], lines[i, :, 1])


def procrustes(lines, e):
    # 线数量
    n = lines.shape[0]
    # 点数量
    m = lines.shape[1]

    data = np.copy(lines)

    remove_translate_all(data)
    remove_scale_all(data)

    ref_shape = np.copy(data[0])
    error = math.inf
    while error > e:
        align_all(ref_shape, data)
        center = mean_shape(data)
        error = procrustes_dis(ref_shape, center)
        ref_shape = center
        print(error)

    return data, ref_shape


def main():
    lines = read_data("data")

    plt.subplot(2, 2, 1)
    # plt.xlim(-30, 30)
    # plt.ylim(-30, 30)
    plt.axis("equal")
    plot(lines, -1)

    data, center = procrustes(lines, 0.001)
    plt.subplot(2, 2, 2)
    plt.axis("equal")
    plot(data, -1)

    plt.subplot(2, 2, 3)
    plt.axis("equal")
    plot(np.expand_dims(center, 0), 1)

    plt.show()


if __name__ == "__main__":
    main()
