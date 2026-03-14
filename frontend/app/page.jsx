'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Typography, message } from 'antd'
import { decodeApiBaseFromQuery, saveApiBase } from '../lib/api'

const { Title, Paragraph } = Typography

const PROMPT_EXAMPLES = [
  {
    title: '生圖優化示例 1',
    before: '我想要畫一隻貓',
    after: `a cute cat in warm sunset light, golden hour atmosphere, soft cinematic lighting, expressive shining eyes, adorable small movement like stretching or tilting its head, warm emotional storytelling scene, soft orange and golden colour palette, gentle rim lighting from sunset, shallow depth of field, highly detailed fur texture, peaceful evening mood, dreamy warm ambience, ultra detailed, professional photography style, soft background bokeh, emotional and heartwarming composition`
  },
  {
    title: '對話/寫作優化示例',
    before: '我想要一個論文題目',
    after: `你是一位學術寫作導師與研究方法顧問，專長於研究流程設計、文獻整合與引用審校，主要協助需要發展論文題目的研究生或學者。本次對話的目標是幫助使用者發展一個具有競賽潛力的論文題目，因此回應時需同時考量學術嚴謹性、研究創新性與實際應用價值。請以專業且具啟發性的方式回應，每次回答都應提供具體建議與引導性問題，以協助使用者逐步釐清研究方向與研究設計。

所有回覆需使用繁體中文，並遵循「先給結論，再補充必要說明」的原則。若使用者提供的資訊不足，請先列出「待確認資訊」，說明目前缺乏哪些關鍵資料，而不是直接推測使用者背景或研究條件。你的角色是協助使用者建立清晰且可行的研究路徑，因此應主動引導其思考研究問題、研究範圍與研究目標。

在研究流程上，請優先協助使用者界定研究問題，明確說明研究主題、研究範圍與研究目的，並進一步思考研究能產生的學術貢獻與驗收標準。接著，需協助整理可能的資料來源與證據強度，建議優先參考學術資料庫，例如 Scopus、Web of Science、Google Scholar、JSTOR、IEEE 或 PubMed 等。分析資料時，應清楚區分哪些內容屬於經驗證據（empirical evidence），哪些屬於學術推論（scholarly interpretation），並指出目前研究領域中的研究缺口（research gap）。

在確立研究問題後，請分析當前學術界的熱門研究趨勢與發展方向，評估使用者的研究興趣是否與現有研究前沿相契合，並思考是否存在跨領域研究的可能性。根據上述分析，協助使用者提出數個具有潛力的論文題目候選，並從創新性、可行性、學術價值與實務應用價值等面向進行評估，說明每個題目的優勢與可能限制。

在題目初步確定後，請建議適合的研究架構與論文結構。一般建議採用 IMRaD（Introduction、Methods、Results、Discussion） 或其他等價研究框架，並說明每一部分應包含的核心內容，例如研究背景、研究方法、資料來源、分析策略與可能的研究結果。隨後可進一步建立初步的論文大綱，將整篇論文分為若干段落，每一段需明確指出該段的核心主張以及支持該主張的證據或資料來源。

若該論文是為了參與學術競賽，則需同時考慮競賽的評分標準與評審關注重點。請協助分析題目的創新程度、研究設計的嚴謹性、研究成果的社會或產業影響，以及是否具有跨領域整合的潛力。必要時可提出強化競賽競爭力的策略，例如強化研究方法、增加資料可信度或提升研究成果的應用價值。

在整個對話過程中，請持續提出能促進思考的引導性問題，例如使用者的研究領域與專長是什麼、目前關注的學術議題有哪些、是否已有特定理論或研究方法想採用、目標競賽的名稱與評分標準為何，以及是否已有初步文獻或資料來源。若使用者提出的題目存在概念模糊、研究範圍過大或缺乏創新性的問題，請提供建設性的修正建議，並提出替代研究方向，同時鼓勵使用者從多角度思考研究可能性。

整體而言，你的目標是協助使用者發展一個符合學術研究標準、具有創新性與可行研究設計，並且具備競賽潛力的論文題目，使該題目能引起學術界的興趣並具有實際研究價值。`
  },
  {
    title: '影片/動畫優化示例',
    before: '我想要畫一個小企鵝在敲門',
    after: `A cute little penguin standing in front of a small wooden door in a snowy environment.
The penguin gently knocks on the door with its flipper, looking curious and slightly shy.
The scene takes place during a soft winter afternoon with warm light coming from inside the house.
Snow slowly falls in the background, creating a peaceful atmosphere.

The penguin tilts its head slightly while knocking again, waiting patiently.
The camera slowly pushes in toward the penguin, focusing on its expressive eyes and adorable movement.

Style: cinematic animation, warm and heart-warming mood, soft lighting, high detail, smooth motion.
Environment: snowy landscape, small cosy house, wooden door with warm light glowing from inside.
Camera: slow push-in shot, shallow depth of field, gentle cinematic framing.

Quality: high detail, natural motion, clean animation, 4K cinematic quality.
Duration: 5–8 seconds.

Negative: blurry, distorted, deformed penguin, low resolution, flickering, unnatural motion, text, watermark.`
  }
]

export default function HomePage() {
  const router = useRouter()

  useEffect(() => {
    // 解析 ?api=base64(url) 並覆蓋後端位址
    const params = new URLSearchParams(window.location.search)
    const encoded = params.get('api')
    if (encoded) {
      const decoded = decodeApiBaseFromQuery(encoded)
      if (decoded) {
        saveApiBase(decoded)
        message.success('已切換後端地址')
      } else {
        message.error('後端地址解析失敗')
      }
      router.replace('/')
    }
  }, [router])

  const openStartPanel = () => {
    router.push('/start')
  }

  return (
    <div className="landing-shell">
      <header className="landing-nav">
        <div className="brand-wrap">
          <span className="brand-mark" />
          <span className="brand-text">需求羅盤</span>
        </div>
        <nav className="nav-links">
          <a href="#feature">功能</a>
          <button type="button" className="nav-link-btn" onClick={openStartPanel}>開始</button>
          <button type="button" className="nav-link-btn" onClick={openStartPanel}>建立專案</button>
        </nav>
      </header>

      <section className="hero-grid">
        <div className="hero-copy">
          <p className="hero-kicker">Prompt Design Studio</p>
          <h1>三步驟，讓你更快知道自己要什麼</h1>
          <p className="hero-sub">
            先輸入你的想法，我們會用有引導性的提問幫你收斂需求，最後輸出可直接使用的提示詞。
          </p>
          <div className="hero-badges">
            <span>輸入想法</span>
            <span>AI 追問</span>
            <span>完整需求</span>
          </div>
          <div className="hero-actions">
            <button type="button" className="cta-solid" onClick={openStartPanel}>立即開始</button>
            <a href="#feature" className="cta-outline">先看功能</a>
          </div>
        </div>
        <div className="hero-art">
          <div className="art-panel">
            <div className="art-header" />
            <div className="art-row">
              <div className="art-chart" />
              <div className="art-donut" />
            </div>
            <div className="art-cards">
              <span />
              <span />
              <span />
            </div>
            <i className="floating-arrow" />
            <i className="floating-coin" />
          </div>
        </div>
      </section>

      <section id="feature" className="feature-grid">
        <article>
          <h3>用戶語言</h3>
          <p>不用技術術語。直接說「你想做到什麼」，系統會幫你拆成可執行需求。</p>
        </article>
        <article>
          <h3>動態追問</h3>
          <p>問題會根據你的回答調整，逐步把模糊想法收斂成明確方案。</p>
        </article>
        <article>
          <h3>可直接投餵</h3>
          <p>最後輸出模型可讀的完整提示詞，不是空泛建議。</p>
        </article>
      </section>

      <section className="example-section">
        <h2>提示詞優化前後示例</h2>
        <p>往下看三個真實範例：從一句模糊需求，變成可直接投餵模型的高品質提示詞。</p>
        <div className="example-grid">
          {PROMPT_EXAMPLES.map((item) => (
            <article key={item.title} className="example-card">
              <h3>{item.title}</h3>
              <div className="example-block before">
                <label>優化前</label>
                <pre>{item.before}</pre>
              </div>
              <div className="example-block after">
                <label>優化後</label>
                <pre>{item.after}</pre>
              </div>
            </article>
          ))}
        </div>
      </section>

    </div>
  )
}
