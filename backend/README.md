# 需求羅盤 Backend（FastAPI）

## 快速啟動（本機）
1. 建立虛擬環境並安裝依賴
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. 設定環境變數（複製 `.env.example` 為 `.env`，填入 OpenAI 或 Qwen 金鑰）

3. 啟動服務
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## 主要路由（與原版相容）
- `GET /api/health`
- `POST /api/generate-questions`
- `POST /api/submit-answers`
- `POST /api/continue-with-feedback`
- `POST /api/append-questions`
- `GET /api/session/{session_id}`
- `GET /api/session/{session_id}/rounds`
- `POST /api/generate-pdf`
- `GET /api/download-pdf/{session_id}`
- `DELETE /api/session/{session_id}`

## 設定重點
- `.env` 內容可參考 `.env.example`
- `DAILY_TOKEN_LIMIT=0` 代表不限制
- `DB_PATH` 可設為 `data/requirement_compass.db`（`docker-compose` 預設）

## 補充說明
- 若沒有 API 金鑰，後端會使用本機 stub 產生示例問題與報告。
- 資料庫預設為 `requirement_compass.db`（Docker 情境下可用 `/app/data/requirement_compass.db`）。
