import os
from main_ir import ObjectDetectionV2IR


if __name__ == "__main__":
    model_path = r'D:\AI_Vision_Project\Bug\20230808_Test04\Feature Detection5 meta Tool1\Models\Model_2023_08_08_1\model_final.xml'
    data_path = r'D:\AI_Vision_Project\Bug\20230808_Test04\Feature Detection5 meta Tool1\Images'
    device = "GPU"  # GPU,CPU
    InsSeg = ObjectDetectionV2IR(model_path, device)

    for i, file in enumerate(os.listdir(data_path)):
        if (file[-4:] != '.png' and file[-4:] != '.jpg' and file[-4:] != '.bmp'):
            continue
        img_path = os.path.join(data_path, file)
        result = InsSeg(img_path)
        out_path = os.path.join(os.path.dirname(model_path), 'ov')
        os.makedirs(out_path, exist_ok=True)
        InsSeg.draw_detections(img_path, result, out_path)
