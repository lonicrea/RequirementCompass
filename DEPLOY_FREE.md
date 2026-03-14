# 免費部署方案（Vercel + Cloudflare Tunnel）

這套方案可在不綁卡的情況下，先把網站公開給其他人使用。

- 前端：Vercel（免費）
- 後端：你的本機 FastAPI（經由 Cloudflare Tunnel 公網化）

> 注意：後端跑在你的電腦，所以電腦必須保持開機且網路穩定。

---

## 1) 啟動後端（本機）

在 `backend/`：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

填好 `.env` 的 `OPENAI_API_KEY`，再啟動：

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

健康檢查：

```bash
http://localhost:5000/api/health
```

---

## 2) 建立 Cloudflare Tunnel（把本機後端變公網）

先安裝 `cloudflared`，然後登入：

```bash
cloudflared tunnel login
```

建立 tunnel：

```bash
cloudflared tunnel create requirement-compass-api
```

新增 DNS（假設你的網域是 `example.com`）：

```bash
cloudflared tunnel route dns requirement-compass-api api.example.com
```

把 `backend/cloudflared.example.yml` 複製成 `backend/cloudflared.yml`，把 `<YOUR_TUNNEL_UUID>`、`api.example.com` 改成你的值。

啟動 tunnel：

```bash
cloudflared tunnel --config backend/cloudflared.yml run
```

此時你的 API 公網位址是：

```text
https://api.example.com/api
```

---

## 3) 部署前端到 Vercel（免費）

### 方式 A：Vercel 網站

1. 到 Vercel 匯入 repo：`lonicrea/RequirementCompass`
2. Root Directory 選：`frontend`
3. Environment Variable 設定：
   - `NEXT_PUBLIC_API_BASE_URL=https://api.example.com/api`
4. Deploy

### 方式 B：CLI

在 `frontend/`：

```bash
npm i -g vercel
vercel
```

部署後到 Vercel 專案設定補上：

```text
NEXT_PUBLIC_API_BASE_URL=https://api.example.com/api
```

然後重新部署一次。

---

## 4) 上線驗證

1. 打開 Vercel 前端網址
2. 輸入需求並提交
3. 檢查瀏覽器 Network，請求是否打到 `https://api.example.com/api/*`

---

## 常見問題

- **CORS 錯誤**：確認後端有啟動且 tunnel 正常；本專案預設 CORS 全開。
- **超時**：你的本機網路太慢或後端沒開；先看 `backend` terminal 是否有請求進來。
- **他人打不開**：你的 tunnel 沒在跑，或 DNS 還沒生效。

