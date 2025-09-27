# -*- coding: utf-8 -*-
import os
import argparse

def rename_jpeg_to_jpg(folder_path):
    """
    将文件夹中所有.jpeg扩展名的文件重命名为.jpg扩展名

    Args:
        folder_path (str): 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的文件夹")
        return

    # 统计重命名的文件数量
    renamed_count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理文件，跳过子文件夹
        if os.path.isfile(file_path):
            # 检查文件扩展名是否为.jpeg（不区分大小写）
            if filename.lower().endswith('.jpeg'):
                # 生成新的文件名，将.jpeg替换为.jpg
                new_filename = filename[:-5] + '.jpg'  # 去掉.jpeg(5个字符)加上.jpg
                new_file_path = os.path.join(folder_path, new_filename)

                # 检查是否已经存在同名的.jpg文件
                if os.path.exists(new_file_path):
                    print(f"警告: 文件 '{new_filename}' 已存在，跳过 '{filename}'")
                    continue

                # 重命名文件
                try:
                    os.rename(file_path, new_file_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"错误: 无法重命名 '{filename}': {str(e)}")

    print(f"\n完成! 共重命名了 {renamed_count} 个文件")

def main():
    parser = argparse.ArgumentParser(description='将文件夹中所有.jpeg文件重命名为.jpg格式')
    parser.add_argument('--folder_path', default="../imgs3", help='要处理的文件夹路径')

    args = parser.parse_args()

    print(f"开始处理文件夹: {args.folder_path}")
    rename_jpeg_to_jpg(args.folder_path)

if __name__ == "__main__":
    main()
