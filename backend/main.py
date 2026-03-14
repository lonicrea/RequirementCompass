import uvicorn
from app import create_app

app = create_app()


if __name__ == "__main__":
    # 預設關閉自動重載，避免 Windows 上出現重複監聽程序。
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
