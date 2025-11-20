import time
from math import floor

import cv2
import numpy as np

from utils.utils import img_bit_decomposition, sine_cubic_map_sequence, logistic_map, logistics_map_with_direction, \
    restore_sequence_optimized, get_image_hash, split_hash_into_blocks

img_path_base = "../img/"
img_name = "House_jiamied.tiff"  # 自行修改图片名，默认为 lena 图
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_jiemi.png"


x01 = 0.3 #0.3000000000000001
omega01 = 2.4 #2.1

x02 = 0.5 #0.50000001
r02 = 3.95 #3.950000000000001

def reconstruct_image(bit_planes):
    """Reconstruct a single-channel image from bit planes"""
    return np.sum(bit_planes * (2 ** np.arange(8)[:, None, None]), axis=0).astype(np.uint8)


def img_bit_decomposition_color(img):
    """Decompose a color image into bit planes"""
    # 对每个通道进行位平面分解
    return [img_bit_decomposition(img[:, :, channel]) for channel in range(3)]


def reconstruct_image_color(bit_planes_color):
    """Reconstruct a color image from bit planes"""
    channels = [reconstruct_image(bit_planes) for bit_planes in bit_planes_color]
    return cv2.merge(channels)

def rotate_bit_plane(bit_plane, angle):
    '''对位平面进行旋转'''
    if angle == 90:
        return cv2.rotate(bit_plane, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 解密时旋转相反方向
    elif angle == 180:
        return cv2.rotate(bit_plane, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(bit_plane, cv2.ROTATE_90_CLOCKWISE)
    else:
        return bit_plane  # 如果角度不是90, 180, 270，则不做旋转


def calculate_sums(blocks):
    # 使用 numpy 进行高效的切片和求和操作
    total = np.sum(blocks)
    sum1 = np.sum(blocks[:16])
    sum2 = np.sum(blocks[16:32])
    sum3 = np.sum(blocks[32:48])
    sum4 = np.sum(blocks[48:64])
    sum5 = np.sum(blocks[:33])  # 包含前33块

    return sum1, sum2, sum3, sum4, sum5, total


# 计算偏移量
def calculate_bias(sums, total):
    return [s * 1e-2 / total for s in sums]  # 在计算偏移量时直接乘以 10^-2

def jiami(img):
    # 1 开始计时
    start_time_all = time.time()
    # 读取彩色图像
    # sha512_hash = "66e4dc62a945f26fd1b5da9146a38efef51fc1f983e82bafc2d1200397631c1ab729da12bad5b75da978bfa90671d4fec49e7a604513d2a6eba8682527466f24"
    sha512_hash = "f7028c87feb3cc1bf9ef6e69bc954104b5eea6c34fc0f923fb286c9f467a11ac2ef923a4a0fc0712c230ae3f4bc7c75994879e7942c56b419318674e0790710c"
    print(sha512_hash)
    blocks = split_hash_into_blocks(sha512_hash)
    # 计算每个部分的和
    sum1, sum2, sum3, sum4, sum5, total = calculate_sums(blocks)

    # 计算每个部分的偏移量
    biases = calculate_bias([sum1, sum2, sum3, sum4, sum5], total)

    # 1-0 结束计时
    end_time = time.time()

    # 1-1 计算并输出消耗时间
    elapsed_time = end_time - start_time_all
    print(f"1111: {elapsed_time:.4f} 秒")

    # 提前计算 x0
    x0_1 = float(0.5 + biases[1])
    # x0_1 = 0.5023017931656705
    # print(f"x0_1: {x0_1}")
    x0_2 = float(0.5 + biases[2])
    # x0_2 = 0.5026717040712755
    # print(f"x0_2: {x0_2}")
    x0_3 = float(0.5 + biases[3])
    # x0_3 = 0.5020705988496673
    # print(f"x0_3: {x0_3}")
    x0_4 = float(x02 + biases[4])
    # x0_4 = 0.5054640802977332
    # print(f"x0_4: {x0_4}")

    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)

    m, n, _ = example_img_color.shape

    # 1 开始计时
    start_time = time.time()

    x0_0 = float(x01 + biases[0])

    N0 = 1000
    sine_cubic_map_x0 = sine_cubic_map_sequence(x0_0, omega01, N0 + 3 * m * n)[N0:]

    # 1 结束计时
    end_time = time.time()

    # 1 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生sine_cubic_map_x0时间: {elapsed_time:.4f} 秒")

    # 1-1 开始计时
    start_time = time.time()

    N0 = 1000
    steps1, directions1 = logistics_map_with_direction(x0_1, 3.9, N0 + m * n)
    indexSequence1 = (steps1[N0:], directions1[N0:])

    steps2, directions2 = logistics_map_with_direction(x0_2, 3.8, N0 + m * n)
    indexSequence2 = (steps2[N0:], directions2[N0:])

    steps3, directions3 = logistics_map_with_direction(x0_3, 3.7, N0 + m * n)
    indexSequence3 = (steps3[N0:], directions3[N0:])

    # 1-1 结束计时
    end_time = time.time()

    # 1-1 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生logistics_map_with_direction_map_x0时间: {elapsed_time:.4f} 秒")

    # 1-2 开始计时
    start_time = time.time()

    N1 = 1000
    logistic_map_x0 = logistic_map(x0_4, r02, N1 + m * n)[N1:]
    X01 = [floor(j * 1e14) % 256 for j in logistic_map_x0]

    X01_reshape1 = np.mat(X01, dtype=np.uint8).reshape(m, n)

    X01_bitplanes1 = img_bit_decomposition(X01_reshape1)

    # 1-2 结束计时
    end_time = time.time()

    # 1-2 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生logistic_map_x0时间: {elapsed_time:.4f} 秒")

    #2 开始计时
    start_time = time.time()

    X0 = [floor(i * 1e14) % 256 for i in sine_cubic_map_x0]

    #2 结束计时
    end_time = time.time()

    #2 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生X0时间: {elapsed_time:.4f} 秒")

    # 记录开始时间
    start_time3 = time.time()

    # 应用位平面分解于彩色图像
    decomposed_bit_planes_color = img_bit_decomposition_color(example_img_color)

    for i in range(3):
        for j in range(0, 4):
            decomposed_bit_planes_color[i][j] = rotate_bit_plane(decomposed_bit_planes_color[i][j], 90 * (4 - j))

    for i in range(3):
        for j in range(4, 8):
            # 假设你要旋转decomposed_bit_planes_color[i][j]的高位平面（例如第7位）
            decomposed_bit_planes_color[i][j] = rotate_bit_plane(decomposed_bit_planes_color[i][j], 90 * (8 - j))
            # 解密步骤
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j],
                                                               X01_bitplanes1[j, :, :])

    end_time3 = time.time()
    print("逆比特级xor+反转运行时间:", end_time3 - start_time3, "秒")

    # 从位平面重构彩色图像
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)

    #3 开始计时
    start_time = time.time()

    #3 拆分图像并转换为一维数组
    B, G, R = cv2.split(reconstructed_img_color)

    B1 = B.ravel()
    G1 = G.ravel()
    R1 = R.ravel()

    B22 = restore_sequence_optimized(B1, indexSequence1)
    G22 = restore_sequence_optimized(G1, indexSequence2)
    R22 = restore_sequence_optimized(R1, indexSequence3)

    # 3 结束计时
    end_time = time.time()

    # 3 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生all111时间: {elapsed_time:.4f} 秒")

    all = np.concatenate((R22.ravel(), G22.ravel(), B22.ravel())).astype(np.int32)

    # 3 开始计时
    start_time = time.time()
    # 假设 all 是加密后的数组，X0 是密钥数组
    for j in range(2, -1, -1):  # 逆序处理3轮 2, -1, -1
        for i in range(3 * m * n - 1, -1, -1):  # 逆序处理每个元素
            # 如果在第二轮（j == 1），执行交换操作
            if j == 0:
                # 交换前四位和后四位
                current = all[i]
                front = (current >> 4) & 0x0F
                back = current & 0x0F
                swapped = (back << 4) | front
                all[i] = swapped

            # 异或操作逆向
            sum_val = all[i] ^ X0[i]

            # 计算前一个值（prev）
            if i == 0:
                prev = all[-1]  # 原数组的最后一个元素，此时已解密
            else:
                prev = all[i - 1]  # 加密后的前一个元素值

            # 逆向加法操作
            original = (sum_val - prev) % 256
            all[i] = original

    # for i in range(3 * m * n - 1, -1, -1):  # 逆序处理每个元素
    #     # 如果在第二轮（j == 1），执行交换操作
    #     # 交换前四位和后四位
    #     current = all[i]
    #     front = (current >> 4) & 0x0F
    #     back = current & 0x0F
    #     swapped = (back << 4) | front
    #     all[i] = swapped
    #
    #     # 异或操作逆向
    #     sum_val = all[i] ^ X0[i]
    #
    #     # 计算前一个值（prev）
    #     if i == 0:
    #         prev = all[-1]  # 原数组的最后一个元素，此时已解密
    #     else:
    #         prev = all[i - 1]  # 加密后的前一个元素值
    #
    #     # 逆向加法操作
    #     original = (sum_val - prev) % 256
    #     all[i] = original

    all = all.astype(np.uint8)  # 转换回 uint8，确保适用于 OpenCV

    #3 结束计时
    end_time = time.time()

    #3 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生all时间: {elapsed_time:.4f} 秒")

    # 4 开始计时
    start_time = time.time()

    all_split = np.array_split(all, 3)

    B11 = all_split[2]
    G11 = all_split[1]
    R11 = all_split[0]

    B2 = B11.reshape((m, n))
    G2 = G11.reshape((m, n))
    R2 = R11.reshape((m, n))

    E = cv2.merge([B2, G2, R2])

    # 4 结束计时
    end_time = time.time()

    # 4 合并
    elapsed_time = end_time - start_time
    print(f"合并时间: {elapsed_time:.4f} 秒")
    # 记录结束时间all
    end_time_all = time.time()
    print("总运行时间:", end_time_all - start_time_all, "秒")

    cv2.imshow('Encrypted Image', E)
    cv2.imwrite(encrypt_img_path, E)  # 保存图像
    cv2.waitKey(0)

def main():
    # 执行函数
    jiami(img_path)


if __name__ == '__main__':
    main()
