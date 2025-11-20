import hashlib
import numpy as np

# 加载图像文件并计算SHA-512哈希
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    sha512_hash = hashlib.sha512(image_data).hexdigest()  # 获取SHA-512哈希值
    return sha512_hash

# 将SHA-512哈希值分成64个块，每个块8位，并转换为十进制
def split_hash_into_blocks(sha512_hash):
    return np.array([int(sha512_hash[i:i+2], 16) for i in range(0, len(sha512_hash), 2)])

# 计算每个块的和
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
    return [s / total for s in sums]

# 示例：计算图像的SHA-512哈希并拆分成64个块
image_path = '../img/lena.png'  # 修改为图像的路径
sha512_hash = get_image_hash(image_path)
# 66e4dc62a945f26fd1b5da9146a38efef51fc1f983e82bafc2d1200397631c1ab729da12bad5b75da978bfa90671d4fec49e7a604513d2a6eba8682527466f24
# 3c9909afec25354d551dae21590bb26e38d53f2173b8d3dc3eee4c047e7ab1c1eb8b85103e3be7ba613b31bb5c9c36214dc9f14a42fd7a2fdb84856bca5c44c2
print(f"SHA-512哈希值: {sha512_hash}")
blocks = split_hash_into_blocks(sha512_hash)

# 计算每个部分的和
sum1, sum2, sum3, sum4, sum5, total = calculate_sums(blocks)

# 计算每个部分的偏移量
biases = calculate_bias([sum1, sum2, sum3, sum4, sum5], total)

# 输出结果
print("生成的64个块（每块8位，十进制）：")
for i, block in enumerate(blocks):
    print(f"块 {i + 1}: {block}")

print(f"Key1偏移量: {biases[0]}")
print(f"Key2偏移量: {biases[1]}")
print(f"Key3偏移量: {biases[2]}")
print(f"Key4偏移量: {biases[3]}")
print(f"Key5偏移量: {biases[4]}")
print(f"总和: {total}")
