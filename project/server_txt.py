from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# 数据目录：图片 + TXT 标注文件都放这里
# DATA_DIR = "./data"
DATA_DIR = "../imgs3_result"
# DATA_DIR = "/Users/zhangwei/Downloads/车牌检测训练集/val_detect/blue"

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务，用于访问图片
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


@app.get("/files")
def list_files():
    """
    返回数据集中所有图片 + 对应的TXT标注文件
    """
    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    result = []

    for img in images:
        base = os.path.splitext(img)[0]
        txt_file = base + ".txt"
        result.append({
            "image": img,
            "annotation": txt_file if os.path.exists(os.path.join(DATA_DIR, txt_file)) else None
        })

    return result


@app.get("/annotation/{filename}")
def get_annotation(filename: str):
    """
    读取TXT标注文件，直接返回解析后的标注数据
    格式：label x_center y_center width height pt1_x pt1_y pt2_x pt2_y pt3_x pt3_y pt4_x pt4_y
    """
    if not filename.endswith('.txt'):
        return JSONResponse({"annotations": []})

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        return {"annotations": []}

    annotations = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 13:  # label + x + y + w + h + 8个关键点坐标
                continue

            try:
                annotation = {
                    "line_index": line_idx,  # 记录在文件中的行号
                    "label": parts[0],
                    "center_x": float(parts[1]),
                    "center_y": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                    "keypoints": [
                        [float(parts[5]), float(parts[6])],  # 左上
                        [float(parts[7]), float(parts[8])],  # 右上
                        [float(parts[9]), float(parts[10])],  # 右下
                        [float(parts[11]), float(parts[12])]  # 左下
                    ]
                }
                annotations.append(annotation)

            except ValueError as e:
                print(f"解析第{line_idx}行数据出错: {line} - {e}")
                continue

    except Exception as e:
        print(f"读取TXT文件出错 {path}: {e}")
        return {"annotations": []}

    return {"annotations": annotations}


@app.post("/annotation/{filename}")
async def save_annotation(filename: str, data: dict):
    """
    保存标注数据到TXT文件
    """
    if not filename.endswith('.txt'):
        return JSONResponse({"status": "error", "message": "只支持TXT文件格式"})

    path = os.path.join(DATA_DIR, filename)

    try:
        annotations = data.get("annotations", [])
        lines = []

        for ann in annotations:
            # 构造TXT行：label x y w h pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
            line_parts = [
                ann["label"],
                f"{ann['center_x']:.6f}",
                f"{ann['center_y']:.6f}",
                f"{ann['width']:.6f}",
                f"{ann['height']:.6f}"
            ]

            # 添加4个关键点坐标
            for pt in ann["keypoints"]:
                line_parts.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}"])

            lines.append(" ".join(line_parts))

        # 写入文件
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:  # 如果有内容，在末尾添加换行符
                f.write("\n")

        return {"status": "ok", "file": filename, "count": len(annotations)}

    except Exception as e:
        print(f"保存TXT文件出错 {path}: {e}")
        return JSONResponse({"status": "error", "message": f"保存失败: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    import argparse
    #    python server_txt.py --reload

    parser = argparse.ArgumentParser(description="启动TXT标注文件处理服务器")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8001, help="端口号")
    parser.add_argument("--reload", action="store_true", help="开发模式下自动重载")

    args = parser.parse_args()

    uvicorn.run(
        "server_txt:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )