
pip install fastapi uvicorn
uvicorn server:app --reload --port 8000


浏览器访问：

图片地址示例: http://127.0.0.1:8000/static/img1.jpg

文件列表: http://127.0.0.1:8000/files

标注获取: http://127.0.0.1:8000/annotation/img1.json

标注保存: POST http://127.0.0.1:8000/annotation/img1.json