from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, json

# 数据目录：图片 + JSON 都放这里
DATA_DIR = "./data"

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 ["http://127.0.0.1:5500"] 仅允许某前端
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务，用于访问图片
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


@app.get("/files")
def list_files():
    """
    返回数据集中所有图片 + 对应的标注文件
    """
    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    result = []
    for img in images:
        base = os.path.splitext(img)[0]
        json_file = base + ".json"
        result.append({
            "image": img,
            "annotation": json_file if os.path.exists(os.path.join(DATA_DIR, json_file)) else None
        })
    return result


@app.get("/annotation/{filename}")
def get_annotation(filename: str):
    """
    获取某张图片的标注 JSON
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"shapes": []})
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/annotation/{filename}")
async def save_annotation(filename: str, data: dict):
    """
    保存标注 JSON 文件
    """
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "file": filename}


if __name__ == "__main__":
    """
    python server.py
    # 如果server.py在当前目录
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
    # 或者指定模块路径
    uvicorn project.server:app --reload --host 0.0.0.0 --port 8000
    """
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="启动标注服务器")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--reload", action="store_true", help="开发模式下自动重载")

    args = parser.parse_args()

    uvicorn.run(
        "server:app",  # 或 "project.server:app" 根据实际模块路径调整
        host=args.host,
        port=args.port,
        reload=args.reload
    )