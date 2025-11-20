import cv2
from numpy import uint8, zeros, sin, pi, cos
import numpy as np
import hashlib


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
    # 1-1 开始计时
    import time

    """生成步长和方向序列"""
    x = x0
    directions = np.zeros(n, dtype=np.int32)
    steps = np.zeros(n, dtype=np.int32)

    start_time = time.time()
    for i in range(n):
        x = r * x * (1 - x)
        directions[i] = 1 if x > 0.5 else -1
        steps[i] = int(x * 1e14) % n + 1

    # 1 结束计时
    end_time = time.time()

    # 1 计算并输出消耗时间
    elapsed_time = end_time - start_time
    print(f"产生func时间: {elapsed_time:.4f} 秒")

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

# 加载图像文件并计算SHA-512哈希
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    sha512_hash = hashlib.sha512(image_data).hexdigest()  # 获取SHA-512哈希值
    return sha512_hash

# 将SHA-512哈希值分成64个块，每个块8位，并转换为十进制
def split_hash_into_blocks(sha512_hash):
    # 计算SHA-512哈希的每8位，并将其转换为十进制
    blocks = [int(sha512_hash[i:i+2], 16) for i in range(0, len(sha512_hash), 2)]  # 每两个十六进制字符为一个字节（8位）
    return blocks