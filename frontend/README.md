# 需求羅盤 Frontend（Next.js）

## 快速啟動
1. 安裝依賴
```bash
npm install
```

2. 本機開發
```bash
npm run dev -- --hostname 0.0.0.0 --port 5174
```

3. 建置與正式執行
```bash
npm run build
npm start
```

## 設定
- 在 `.env.local` 設定 `NEXT_PUBLIC_API_BASE_URL`（預設為 `http://localhost:5000/api`）。
- 也可在網址加入 `?api=base64(後端位址)`，頁面會自動切換並存入 `localStorage`。

## 流程對齊
- 首頁輸入想法 → `/questions/{id}` 回答問題 → `/results/{id}` 查看報告/下載 → `/overview/{id}` 查看完整歷程。
- API 路徑與原版一致，可直接切換後端。
