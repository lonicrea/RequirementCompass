# 需求羅盤

需求羅盤是需求對齊工具的替代實作版本，採用 FastAPI 後端與 Next.js 前端。

## 專案結構
- `backend/`：FastAPI + SQLite + OpenAI/Qwen 相容介面
- `frontend/`：Next.js + Ant Design，支援可切換後端位址
- `docker-compose.yml`：一鍵啟動前後端服務

## 快速開始
請先閱讀：
- `backend/README.md`
- `frontend/README.md`
- `DEPLOY_FREE.md`（免費公開部署：Vercel + Cloudflare Tunnel）

## 一鍵部署（Render）
本專案已包含 `render.yaml`，可直接用 Blueprint 方式部署前後端。

1. 開啟：
   - `https://render.com/deploy?repo=https://github.com/lonicrea/RequirementCompass`
2. 在 Render 後台填入後端環境變數：
   - `OPENAI_API_KEY`
3. 點擊部署，等待兩個服務完成：
   - `requirement-compass-backend`
   - `requirement-compass-frontend`

部署完成後，直接打開前端網址即可外部訪問。
