import os
import shutil
import glob

"""
将多个文件夹及其子文件夹合并到一个指定的文件夹中
在Windows上通常使用反斜杠 \ 或双反斜杠 \\，在Unix/Linux/Mac上使用正斜杠 /
"""

# mac 文件夹中为什么有'/Volumes/Samsung USB/small/202312small/浙E8191K_0.993.jpg',
# '/Volumes/Samsung USB/small/202312small/._浙E8191K_0.993.jpg'这样的两个路径，实际只有一张图片
# find . -name '._*' -delete

def move_files_to_folder(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 构建源文件的完整路径
            source_file_path = os.path.join(root, file)
            # 构建目标文件的完整路径
            target_file_path = os.path.join(target_folder, file)
            # 移动文件
            shutil.move(source_file_path, target_file_path)


def move_files_to_folder2(source_folder, target_folder):
    """
    将一个源文件夹 (source_folder) 中的所有文件移动到目标文件夹 (target_folder) 中，并在移动过程中将扩展名为 .jpeg 的文件改为 .jpg
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 构建源文件的完整路径
            source_file_path = os.path.join(root, file)
            # 构建目标文件的完整路径，如果文件是.jpeg，则更改为.jpg
            target_file_path = os.path.join(target_folder, file)
            if target_file_path.endswith('.jpeg'):
                target_file_path = target_file_path[:-5] + '.jpg'  # 更改扩展名
            # 移动文件
            shutil.move(source_file_path, target_file_path)


def move_files_to_folder3(source_folder, target_folder):
    """
    处理后保持原目录不变-》支持子文件夹中还有子文件夹的情况-
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        # 构建在目标文件夹中的相对路径
        relative_path = os.path.relpath(root, source_folder)
        # 在目标文件夹中创建相同的子文件夹结构
        target_subfolder = os.path.join(target_folder, relative_path)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        for file in files:
            # 构建源文件的完整路径
            source_file_path = os.path.join(root, file)
            # 构建目标文件的完整路径，如果文件是.jpeg，则更改为.jpg
            target_file_path = os.path.join(target_subfolder, file)
            if target_file_path.endswith('.jpeg'):
                target_file_path = target_file_path[:-5] + '.jpg'  # 更改扩展名

            # 移动文件前确保目标文件夹存在
            target_dir = os.path.dirname(target_file_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # 移动文件
            try:
                shutil.move(source_file_path, target_file_path)
            except FileNotFoundError as e:
                print(f"File not found: {source_file_path}")
            except Exception as e:
                print(f"Error moving file {source_file_path} to {target_file_path}: {e}")


def move_files_to_folder4(source_folder, target_folder):
    """
        处理后使用目标文件夹-》支持子文件夹中还有子文件夹的情况-
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 构建源文件的完整路径
            source_file_path = os.path.join(root, file)
            # 构建目标文件的完整路径，如果文件是.jpeg，则更改为.jpg
            if file.endswith('.jpeg'):
                target_file_name = file[:-5] + '.jpg'  # 更改扩展名
            else:
                target_file_name = file

            target_file_path = os.path.join(target_folder, target_file_name)

            # 处理文件名冲突
            counter = 1
            original_target_file_path = target_file_path
            while os.path.exists(target_file_path):
                target_file_path = os.path.join(target_folder, f"{os.path.splitext(target_file_name)[0]}_{counter}{os.path.splitext(target_file_name)[1]}")
                counter += 1

            # 移动文件
            try:
                shutil.move(source_file_path, target_file_path)
            except FileNotFoundError as e:
                print(f"File not found: {source_file_path}")
            except Exception as e:
                print(f"Error moving file {source_file_path} to {target_file_path}: {e}")

# 示例使用
# move_files_to_folder2('/Volumes/Samsung USB/202312', '/destination_folder')

# 使用示例
# source_folder = r'D:\uchoicepro\202311'  # 源文件夹路径
# target_folder = r'D:\uchoicepro\202311-copy'  # 目标文件夹路径
source_folder = r'/Volumes/Samsung USB/202312/20231231'  # 源文件夹路径
target_folder = r'/Volumes/Samsung USB/202312-copy4'  # 目标文件夹路径

# 整合文件夹
# move_files_to_folder(source_folder, target_folder)
# move_files_to_folder2('source_folder_path', 'target_folder_path')
move_files_to_folder4(source_folder, target_folder)
# 146+166

def rename_files_to_jpg(folder_path):
    """
        将文件夹中的所有文件重命名为.jpg格式
    """
    # print("文件夹路径:", folder_path)
    print(f"Folder path: {folder_path}")
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    # 检查路径是否为文件夹
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a directory.")
        return

    print("开始重命名文件...")
    # 获取文件夹中的所有文件-
    # 默认情况下，glob.glob 会匹配所有文件类型。如果你的文件夹中有子文件夹或特定类型的文件，可以尝试更精确的匹配模式，例如 *.*
    # files = glob.glob(os.path.join(folder_path, '*'))

    # 使用通配符匹配所有 .jpeg 文件，包括子文件夹中的文件
    # files = glob.glob(os.path.join(folder_path, '**', '*.jpeg'), recursive=True)

    # 匹配所有 .jpeg 和 .jpg 文件
    # files = glob.glob(os.path.join(folder_path, '*.jpeg')) + glob.glob(os.path.join(folder_path, '*.jpg'))

    files = glob.glob(os.path.join(folder_path, '*.jpeg'))
    if not files:
        print("No files found in the specified folder.")
        return

    for file_path in files:
        # 获取文件的目录和文件名
        directory, filename = os.path.split(file_path)
        # 获取文件名和扩展名
        name, ext = os.path.splitext(filename)

        # 新的文件名
        new_filename = f"{name}.jpg"
        new_file_path = os.path.join(directory, new_filename)

        # 检查是否已经是 .jpg 文件
        if ext.lower() == '.jpg':
            print(f"File {filename} is already a .jpg file. Skipping.")
            continue

        # 检查新文件名是否存在
        if os.path.exists(new_file_path):
            print(f"File {new_file_path} already exists. Skipping.")
            continue

        # 重命名文件
        try:
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} to {new_file_path}")
        except Exception as e:
            print(f"Failed to rename {file_path} to {new_file_path}: {e}")

    print("重命名完成。")


# 使用示例
# folder_path = 'imgs'   # 替换为你的文件夹路径
folder_path = r'F:/pro/202311-copy'  # 替换为你的文件夹路径
# rename_files_to_jpg(folder_path)
