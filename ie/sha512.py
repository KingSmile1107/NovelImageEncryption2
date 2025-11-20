import hashlib

# 加载图像文件并计算SHA-512哈希
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    sha512_hash = hashlib.sha512(image_data).hexdigest()  # 获取SHA-512哈希值
    return sha512_hash

# 将SHA-512哈希值分成64个块，每个块8位，并转换为十进制
def split_hash_into_blocks(sha512_hash):
    # 计算SHA-512哈希的每8位
    blocks = [int(sha512_hash[i:i+8], 16) for i in range(0, len(sha512_hash), 8)]
    return blocks

# 示例：计算图像的SHA-512哈希并拆分成64个块
image_path = '../img/lena.png'  # 修改为图像的路径
sha512_hash = get_image_hash(image_path)
print(sha512_hash)
blocks = split_hash_into_blocks(sha512_hash)

# 输出每个块的十进制值
print("生成的64个块（每块8位，十进制）：")
for i, block in enumerate(blocks):
    print(f"块 {i+1}: {block}")
