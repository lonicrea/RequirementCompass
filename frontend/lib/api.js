import axios from 'axios'

const fallbackBase = 'http://localhost:5000/api'
const DEFAULT_TIMEOUT_MS = 20000
const SESSION_TIMEOUT_MS = 45000
const LLM_TIMEOUT_MS = 90000
const LOCAL_BASE_MIGRATIONS = [
  {
    test: /^https?:\/\/localhost:5002\/api$/i,
    target: fallbackBase
  },
  {
    test: /^https?:\/\/127\.0\.0\.1:5002\/api$/i,
    target: 'http://127.0.0.1:5000/api'
  }
]

const normalizeBase = (value) => {
  if (typeof value !== 'string') return ''
  const trimmed = value.trim()
  if (!trimmed) return ''
  return trimmed.replace(/\/+$/, '')
}

const decodeMaybeUri = (value) => {
  if (typeof value !== 'string') return ''
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

const migrateLocalBase = (value) => {
  const normalized = normalizeBase(value)
  if (!normalized) return ''
  for (const rule of LOCAL_BASE_MIGRATIONS) {
    if (rule.test.test(normalized)) return rule.target
  }
  return normalized
}

const isLocalBackend = (value) => /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//i.test(String(value || ''))

const isLocalPage = () => {
  if (typeof window === 'undefined') return true
  const host = String(window.location.hostname || '').toLowerCase()
  return host === 'localhost' || host === '127.0.0.1'
}

export const getApiBase = () => {
  const fromEnv = migrateLocalBase(process.env.NEXT_PUBLIC_API_BASE_URL)
  if (fromEnv) return fromEnv

  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('requirement_compass_api')
    const fromStorage = migrateLocalBase(decodeMaybeUri(saved))
    if (fromStorage) {
      if (saved !== fromStorage) {
        localStorage.setItem('requirement_compass_api', fromStorage)
      }
      // 避免線上站點沿用本機 localhost 設定造成連線失敗。
      if (isLocalBackend(fromStorage) && !isLocalPage()) {
        localStorage.removeItem('requirement_compass_api')
      } else {
        return fromStorage
      }
    }

    // 相容舊版鍵名，讀到後會自動遷移到新鍵名。
    const legacy = localStorage.getItem('clarityai_alt_api')
    const fromLegacy = migrateLocalBase(decodeMaybeUri(legacy))
    if (fromLegacy) {
      if (!(isLocalBackend(fromLegacy) && !isLocalPage())) {
        localStorage.setItem('requirement_compass_api', fromLegacy)
        return fromLegacy
      }
    }
  }
  return fallbackBase
}

export const saveApiBase = (apiBase) => {
  if (typeof window === 'undefined') return
  const normalized = migrateLocalBase(decodeMaybeUri(apiBase))
  if (normalized) localStorage.setItem('requirement_compass_api', normalized)
}

export const decodeApiBaseFromQuery = (encoded) => {
  if (!encoded || typeof encoded !== 'string') return ''
  try {
    return normalizeBase(decodeMaybeUri(atob(encoded)))
  } catch {
    return ''
  }
}

const client = (timeoutMs = DEFAULT_TIMEOUT_MS) => axios.create({
  baseURL: String(getApiBase() || ''),
  timeout: timeoutMs,
  headers: { 'Content-Type': 'application/json' }
})

export const api = {
  health: () => client().get('/health'),
  generateQuestions: (idea, profile = {}, customApi) => client(LLM_TIMEOUT_MS).post('/generate-questions', {
    idea,
    user_identity: profile.user_identity,
    language_region: profile.language_region,
    existing_resources: profile.existing_resources,
    custom_api: customApi
  }),
  getSession: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return client(SESSION_TIMEOUT_MS).get(`/session/${id}`)
  },
  getRounds: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return client().get(`/session/${id}/rounds`)
  },
  appendQuestions: (id, instruction = '', customApi) => client(LLM_TIMEOUT_MS).post('/append-questions', { session_id: id, instruction, custom_api: customApi }),
  submitAnswers: (id, answers, customApi) => client(LLM_TIMEOUT_MS).post('/submit-answers', { session_id: id, answers, custom_api: customApi }),
  continueWithFeedback: (id, feedback, customApi) => client(LLM_TIMEOUT_MS).post('/continue-with-feedback', { session_id: id, feedback, custom_api: customApi }),
  naturalizePrompt: (promptText, promptLanguage = '繁體中文', modeHint = '', customApi) =>
    client(LLM_TIMEOUT_MS).post('/naturalize-prompt', {
      prompt_text: promptText,
      prompt_language: promptLanguage,
      mode_hint: modeHint,
      custom_api: customApi
    }),
  generateFinalPrompt: (id, customApi) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return client(LLM_TIMEOUT_MS).post('/generate-final-prompt', {
      session_id: id,
      custom_api: customApi
    })
  },
  generatePdf: (id) => client().post('/generate-pdf', { session_id: id }),
  downloadPdf: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return client().get(`/download-pdf/${id}`, { responseType: 'blob' })
  }
}
