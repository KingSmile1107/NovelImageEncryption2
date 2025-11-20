import time
from math import floor

import cv2
import numpy as np

import cv2
from numpy import uint8, zeros, sin, pi, cos
import numpy as np

def img_bit_decomposition(img):
    '''图像的位平面分解'''
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2**i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r

def logistics_map_with_direction(x0, r, n):
    """生成步长和方向序列"""
    x = x0
    directions = np.zeros(n, dtype=np.int32)
    steps = np.zeros(n, dtype=np.int32)

    for i in range(n):
        x = r * x * (1 - x)
        directions[i] = 1 if x > 0.5 else -1
        steps[i] = int(x * 1e14) % n + 1

    return steps, directions
def get_shuffle_indices_optimized(n, steps, directions):
    """优化后的索引生成函数"""
    # 创建一个数组来存储可用的位置
    available = np.arange(n, dtype=np.int32)
    available_count = n
    indices = np.zeros(n, dtype=np.int32)

    for i in range(n):
        # 计算当前位置
        step = steps[i] % available_count

        if directions[i] < 0:
            step = available_count - step
        if step >= available_count:
            step = step % available_count

        # 获取并移除选中的位置
        selected_index = available[step]
        indices[i] = selected_index

        # 更新可用位置数组（通过将选中位置与最后一个位置交换，然后缩小数组）
        available[step] = available[available_count - 1]
        available_count -= 1

    return indices


def shuffle_sequence_optimized(sequence, indexSequence):
    """优化后的序列置乱函数"""
    n = len(sequence)
    steps, directions = indexSequence

    indices = get_shuffle_indices_optimized(n, steps, directions)

    shuffled_sequence = sequence[indices]

    return shuffled_sequence


def restore_sequence_optimized(shuffled_sequence, indexSequence):
    """优化后的序列恢复函数"""
    steps, directions = indexSequence
    n = len(shuffled_sequence)
    indices = get_shuffle_indices_optimized(n, steps, directions)

    restore_indices = np.zeros_like(indices)
    restore_indices[indices] = np.arange(n)

    restored_sequence = shuffled_sequence[restore_indices]

    return restored_sequence

def logistic_map(x, r=3.9999, sequence_length=10):
    '''Logistic映射函数
    x: 当前值
    r: 控制参数
    sequence_length: 生成的序列长度
    '''
    sequence = [x]
    for _ in range(sequence_length - 1):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence

def sine_cubic_map(x, omega):
    return (np.sin(omega * (10 - x) / (x ** 3 - x))) % 1

def sine_cubic_map_sequence(x, omega, sequence_length=10):
    """生成 sine_cubic_map 的序列"""
    sequence = []
    for _ in range(sequence_length):
        x = sine_cubic_map(x, omega)
        sequence.append(x)
    return np.array(sequence)
img_path_base = "../img/"
img_name = "lena.png"  # 自行修改图片名，默认为 lena 图
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_jiamied.png"

x01 = 0.3
omega01 = 2.4 # lena entropy 15.915

x02 = 0.5
r02 = 3.95

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
    # 使用cv2.rotate进行旋转
    if angle == 90:
        return cv2.rotate(bit_plane, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(bit_plane, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(bit_plane, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return bit_plane  # 如果角度不是90, 180, 270，则不做旋转

def jiami(img):
    # 1 开始计时
    start_time_all = time.time()
    # 读取彩色图像
    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)
    m, n, _ = example_img_color.shape

    #1 开始计时
    start_time = time.time()

    N0 = 1000
    sine_cubic_map_x0 = sine_cubic_map_sequence(x01, omega01, N0 + 3 * m * n)[N0:]

    #1 结束计时
    end_time = time.time()

    #1 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生sine_cubic_map_x0时间: {elapsed_time:.4f} 秒")

    # 1-1 开始计时
    start_time = time.time()

    N0 = 1000
    steps1, directions1 = logistics_map_with_direction(0.5, 3.9, N0 + m * n)
    indexSequence1 = (steps1[N0:], directions1[N0:])

    steps2, directions2 = logistics_map_with_direction(0.5, 3.8, N0 + m * n)
    indexSequence2 = (steps2[N0:], directions2[N0:])

    steps3, directions3 = logistics_map_with_direction(0.5, 3.7, N0 + m * n)
    indexSequence3 = (steps3[N0:], directions3[N0:])

    # 1-1 结束计时
    end_time = time.time()

    # 1-1 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生logistics_map_with_direction_map_x0时间: {elapsed_time:.4f} 秒")

    # 1-2 开始计时
    start_time = time.time()

    N1 = 1000
    logistic_map_x0 = logistic_map(x02, r02, N1 + m * n)[N1:]
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

    #3 开始计时
    start_time = time.time()

    #3 拆分图像并转换为一维数组
    B, G, R = cv2.split(example_img_color)
    all = np.concatenate((R.ravel(), G.ravel(), B.ravel())).astype(np.int32)

    for j in range(3):
        for i in range(3 * m * n):
            if i == 0:
                all[i] = ((all[i] + all[-1]) % 256) ^ X0[i]
            else:
                all[i] = ((all[i] + all[i - 1]) % 256) ^ X0[i]

            if j == 0:
                # 使用位运算来交换前四位和后四位
                front = (all[i] >> 4) & 0x0F  # 提取前四位
                back = all[i] & 0x0F  # 提取后四位

                # 交换位置并组合成新的8位数
                all[i] = (back << 4) | front

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

    B22 = shuffle_sequence_optimized(B11, indexSequence1)
    G22 = shuffle_sequence_optimized(G11, indexSequence2)
    R22 = shuffle_sequence_optimized(R11, indexSequence3)

    # 4 结束计时
    end_time = time.time()

    # 4 合并
    elapsed_time = end_time - start_time
    print(f"置乱: {elapsed_time:.4f} 秒")

    B2 = B22.reshape((m, n))
    G2 = G22.reshape((m, n))
    R2 = R22.reshape((m, n))

    E = cv2.merge([B2, G2, R2])

    # 4 结束计时
    end_time = time.time()

    # 4 合并
    elapsed_time = end_time - start_time
    print(f"合并时间: {elapsed_time:.4f} 秒")

    start_time5 = time.time()
    # 应用位平面分解于彩色图像
    decomposed_bit_planes_color = img_bit_decomposition_color(E)

    for i in range(3):
        for j in range(4, 8):
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j],
                                                               X01_bitplanes1[j, :, :])
            # 假设你要旋转decomposed_bit_planes_color[i][j]的高位平面（例如第7位）
            decomposed_bit_planes_color[i][j] = rotate_bit_plane(decomposed_bit_planes_color[i][j], 90 * (8 - j))

    for i in range(3):
        for j in range(0, 4):
            decomposed_bit_planes_color[i][j] = rotate_bit_plane(decomposed_bit_planes_color[i][j], 90 * (4 - j))

    # 从位平面重构彩色图像
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)

    # 记录结束时间
    end_time5 = time.time()
    print("比特级xor+反转运行时间:", end_time5 - start_time5, "秒")

    # 记录结束时间all
    end_time_all = time.time()
    print("总运行时间:", end_time_all - start_time_all, "秒")

    cv2.imwrite(encrypt_img_path, reconstructed_img_color)  # 保存图像

def main():
    # 执行函数
    jiami(img_path)


if __name__ == '__main__':
    main()
