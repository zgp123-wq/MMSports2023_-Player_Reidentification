import os

def count_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 假设这是常见的图片文件扩展名列表

    image_count = 0

    for filename in os.listdir(folder_path):
        _, extension = os.path.splitext(filename)
        if extension.lower() in image_extensions:
            image_count += 1

    return image_count

# 替换下面的路径为您要统计图片数量的文件夹路径
folder_path_to_count = '/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/data_reid/reid_challenge/gallery'
image_count_result = count_images_in_folder(folder_path_to_count)

print(f"文件夹 '{folder_path_to_count}' 中的图片数量为: {image_count_result}")
