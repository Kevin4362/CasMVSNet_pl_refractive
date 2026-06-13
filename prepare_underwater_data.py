import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# ================= 用户配置（请根据实际修改）=================
ORIG_IMG_DIR = "/mnt/d/Thesis/Mermaid/SR202204_LDM-S_D01"  # 存放原始图片（文件名与 XML label 一致，例如 SR202204_LDM-S_D01_0000.jpg）
XML_FILE = "camera_poses.xml"  # 相机轨迹 XML 文件
OUTPUT_DIR = "my_test_data"  # 输出目录（将创建 images/ 和 cams/）

# 图像尺寸
W = 3840
H = 2880

# 相机内参（来自标定，单位：像素）
fx = 2334.29
fy = 2334.29  # 假设 fx == fy
cx = W / 2 + (-12.752)  # 1907.248
cy = H / 2 + (-16.6962)  # 1423.3038

# 畸变系数 (k1, k2, p1, p2, k3) 顺序符合 OpenCV
dist_coeffs = np.array([-0.222446, -0.310621, -0.000995472, -7.8498e-05, -0.0835057], dtype=np.float32)

# 深度范围（占位，可根据场景调整）
depth_min = 500.0  # 单位 mm，需根据实际物体距离估计
depth_interval = 2.65  # DTU 默认值，可先保留

# 配对策略：每个参考图选多少个源视图（推荐 4 或 5）
n_src_views = 4  # 前后各 2 张，共 4 张源图（加上参考图共 5 视图）


# ===========================================================

def parse_xml(xml_path):
    """解析 XML，返回列表 [(label, transform_4x4), ...]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cameras = root.findall("camera")
    camera_list = []
    for cam in cameras:
        label = cam.get("label")
        transform_str = cam.find("transform").text.strip()
        vals = list(map(float, transform_str.split()))
        transform_mat = np.array(vals).reshape(4, 4)  # 世界 → 相机
        camera_list.append((label, transform_mat))
    return camera_list


def generate_pair_txt(num_images, output_path, n_src):
    """
    n_src: 期望的源视图数量（包括前后，不含参考图本身）
    对于线性序列，直接截断边界，可能实际源视图数少于 n_src。
    """
    with open(output_path, 'w') as f:
        f.write(f"{num_images}\n")
        half = n_src // 2
        for ref in range(num_images):
            src_list = []
            # 向前取 half 张，向后取 half 张（若 n_src 奇数，向前多取一张）
            for offset in range(-half, half + 1):
                if offset == 0:
                    continue
                src_id = ref + offset
                if 0 <= src_id < num_images:
                    src_list.append(src_id)
            # 去重（理论上无重复）
            src_list = list(dict.fromkeys(src_list))
            # 如果实际数量不足 n_src，可以保持原样（模型通常接受可变数量）
            f.write(f"{ref}\n")
            f.write(f"{len(src_list)} " + " ".join(map(str, src_list)) + "\n")


def main():
    # 创建输出目录
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "cams"), exist_ok=True)

    # 解析 XML
    camera_list = parse_xml(XML_FILE)
    print(f"找到 {len(camera_list)} 个相机位姿")

    # 去畸变映射表（基于原始内参和畸变系数）
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    # 获取理想内参（去畸变后，所有像素有效）
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (W, H), 1, (W, H))
    print("去畸变后理想内参:\n", new_camera_matrix)
    # 想知道去畸变后的图像的（x，y）处的像素，就通过map去找他对应在原图中的那个像素的颜色即可
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (W, H), cv2.CV_32FC1)

    # 处理每一张图像
    for idx, (label, ext_mat) in enumerate(camera_list):
        # 查看待处理的图片文件夹中的图片是否存在
        src_img_path = os.path.join(ORIG_IMG_DIR, f"{label}.jpg")
        if not os.path.exists(src_img_path):
            print(f"警告: 图片 {src_img_path} 不存在，跳过")
            continue

        # 查看图片是否正确被读取到
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"无法读取 {src_img_path}")
            continue
        # 将图片去畸变
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # 将去畸变的图片进行保存 保存为 8 位数字名
        out_img_name = f"{idx:08d}.jpg"
        out_img_path = os.path.join(OUTPUT_DIR, "images", out_img_name)
        cv2.imwrite(out_img_path, undistorted)
        print(f"已生成: {out_img_path}")

        # 生成对应的 cam 文件
        cam_file_path = os.path.join(OUTPUT_DIR, "cams", f"{idx:08d}_cam.txt")
        with open(cam_file_path, 'w') as f:
            # 外参矩阵 4x4 (世界 -> 相机)
            for r in range(4):
                f.write(" ".join(f"{ext_mat[r, c]:.12e}" for c in range(4)) + "\n")
            f.write("\n")  # 空行
            # 内参矩阵 3x3 (去畸变后的理想内参)
            for r in range(3):
                f.write(" ".join(f"{new_camera_matrix[r, c]:.6f}" for c in range(3)) + "\n")
            f.write("\n")  # 空行
            f.write(f"{depth_min} {depth_interval}\n")
        print(f"已生成: {cam_file_path}")

    # 生成 pair.txt
    pair_path = os.path.join(OUTPUT_DIR, "cams", "pair.txt")
    generate_pair_txt(len(camera_list), pair_path, n_src_views)
    print(f"已生成配对文件: {pair_path} (每个参考图使用 {n_src_views} 个源视图)")

    print("\n✅ 所有数据准备完成！")
    print(f"去畸变后的图像位于: {OUTPUT_DIR}/images/")
    print(f"相机参数文件位于: {OUTPUT_DIR}/cams/")
    print("下一步，运行 CasMVSNet 测试命令。")


if __name__ == "__main__":
    main()
