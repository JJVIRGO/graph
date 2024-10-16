import pandas as pd
import tifffile
import numpy as np
import os
import matplotlib.pyplot as plt
def pose():
        # 路径配置
        csv_file_path = 'f:/Xenium_V1_FFPE_wildtype_13_4_months_outs/cells.csv'
        tiff_file_path = 'f:/Xenium_V1_FFPE_wildtype_13_4_months_outs/morphology_focus.ome.tif'
        output_dir = 'datasetWild13_4_focus1'


        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 读取CSV文件
        cells_df = pd.read_csv(csv_file_path)

        # 读取3D TIFF图像
        tiff_image = tifffile.imread(tiff_file_path)

        # 获取图像尺寸
        image_shape = tiff_image.shape

        # 计算比例因子
        max_x_csv = cells_df['x_centroid'].max()
        max_y_csv = cells_df['y_centroid'].max()
        x_scale = (image_shape[1]-45) / max_x_csv  #x减小视角窗口往左
        y_scale = (image_shape[0]-40) / max_y_csv  #y变大视角窗口往下
        print(image_shape[1],x_scale,image_shape[0],y_scale)
        # 应用比例因子将微米单位坐标转换为像素单位
        cells_df['x_pixel'] = cells_df['x_centroid'] * x_scale
        cells_df['y_pixel'] = cells_df['y_centroid'] * (y_scale)
        cells_df['cell_area'] = cells_df['cell_area'] * (y_scale)* x_scale/3

        # 提取单个细胞的函数
        def extract_cell(image, x_pixel, y_pixel, area, buffer=10):
            y, x = image.shape
            radius = int(np.sqrt(area / np.pi)) + buffer  # Estimate radius with some buffer

            x_min = max(0, int(x_pixel) - radius)
            x_max = min(x, int(x_pixel) + radius)
            y_min = max(0, int(y_pixel) - radius)
            y_max = min(y, int(y_pixel) + radius)

            cell_image = image[y_min:y_max, x_min:x_max]
            return cell_image

        # 遍历每个细胞并保存其图像

        for index, row in cells_df.iterrows():
            cell_id = row['cell_id']
            x_pixel = row['x_pixel']
            y_pixel = row['y_pixel']
            area = row['cell_area']

            cell_image = extract_cell(tiff_image, x_pixel, y_pixel, area)
            tifffile.imsave(os.path.join(output_dir, f'{cell_id}.tif'), cell_image)

def view_tif_image(file_path):
    image = tifffile.imread(file_path)
    if image.ndim == 3:  # 如果是3D图像，则只取第一个z层
        image = image[6, :, :]
    plt.imshow(image, cmap='gray')
    plt.show()
    print(image.shape)



#pose()
#view_tif_image("datasetWild5_7_focus1/mpeclobl-1.tif")
#view_tif_image("datasetWild5_7_focus1/nbbdfnjb-1.tif")
#view_tif_image("datasetWild5_7_focus1/gfniihik-1.tif")
view_tif_image("f:/10x/Xenium_V1_FFPE_TgCRND8_17_9_months_outs/ddatasetWild5_7_focus1/gfniihik-1.tif")