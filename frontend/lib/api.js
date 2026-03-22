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
  if (/^https?:\/\/[^/]+$/i.test(normalized)) {
    return `${normalized}/api`
  }
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
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('requirement_compass_api')
    const fromStorage = migrateLocalBase(decodeMaybeUri(saved))
    if (fromStorage) {
      if (saved !== fromStorage) {
        localStorage.setItem('requirement_compass_api', fromStorage)
      }
      if (isLocalPage() && !isLocalBackend(fromStorage)) {
        localStorage.removeItem('requirement_compass_api')
      } else if (isLocalBackend(fromStorage) && !isLocalPage()) {
        localStorage.removeItem('requirement_compass_api')
      } else {
        return fromStorage
      }
    }
  }

  if (isLocalPage()) {
    return fallbackBase
  }

  const fromEnv = migrateLocalBase(process.env.NEXT_PUBLIC_API_BASE_URL)
  if (fromEnv) return fromEnv

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

const buildUrl = (path) => `${String(getApiBase() || '')}${path}`

const buildTimeoutError = (timeoutMs) => {
  const error = new Error(`timeout of ${timeoutMs}ms exceeded`)
  error.code = 'ECONNABORTED'
  error.name = 'AxiosError'
  return error
}

const tryParseJson = async (response) => {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

const request = async (path, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(buildUrl(path), {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      }
    })

    const contentType = response.headers.get('content-type') || ''
    const isJson = contentType.includes('application/json')
    const data = isJson ? await response.json() : await tryParseJson(response)

    if (!response.ok) {
      const error = new Error(
        (data && typeof data === 'object' && data.error)
          ? data.error
          : `Request failed with status code ${response.status}`
      )
      error.name = 'AxiosError'
      error.response = { status: response.status, data }
      throw error
    }

    return {
      data,
      status: response.status,
      headers: response.headers,
      config: options
    }
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw buildTimeoutError(timeoutMs)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

const requestBlob = async (path, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(buildUrl(path), {
      ...options,
      signal: controller.signal,
      headers: {
        ...(options.headers || {})
      }
    })

    if (!response.ok) {
      const data = await tryParseJson(response)
      const error = new Error(
        (data && typeof data === 'object' && data.error)
          ? data.error
          : `Request failed with status code ${response.status}`
      )
      error.name = 'AxiosError'
      error.response = { status: response.status, data }
      throw error
    }

    return {
      data: await response.blob(),
      status: response.status,
      headers: response.headers,
      config: options
    }
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw buildTimeoutError(timeoutMs)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  health: () => request('/health'),
  analyzeRequirements: (id, customApi) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return request(
      '/analyze-requirements',
      {
        method: 'POST',
        body: JSON.stringify({
          session_id: id,
          custom_api: customApi
        })
      },
      LLM_TIMEOUT_MS
    )
  },
  generateQuestions: (idea, profile = {}, customApi) => request(
    '/generate-questions',
    {
      method: 'POST',
      body: JSON.stringify({
        idea,
        user_identity: profile.user_identity,
        language_region: profile.language_region,
        existing_resources: profile.existing_resources,
        custom_api: customApi
      })
    },
    LLM_TIMEOUT_MS
  ),
  getSession: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return request(`/session/${id}`, {}, SESSION_TIMEOUT_MS)
  },
  getRounds: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return request(`/session/${id}/rounds`)
  },
  appendQuestions: (id, instruction = '', customApi) => request(
    '/append-questions',
    {
      method: 'POST',
      body: JSON.stringify({ session_id: id, instruction, custom_api: customApi })
    },
    LLM_TIMEOUT_MS
  ),
  submitAnswers: (id, answers, customApi) => request(
    '/submit-answers',
    {
      method: 'POST',
      body: JSON.stringify({ session_id: id, answers, custom_api: customApi })
    },
    LLM_TIMEOUT_MS
  ),
  continueWithFeedback: (id, feedback, customApi) => request(
    '/continue-with-feedback',
    {
      method: 'POST',
      body: JSON.stringify({ session_id: id, feedback, custom_api: customApi })
    },
    LLM_TIMEOUT_MS
  ),
  naturalizePrompt: (promptText, promptLanguage = '繁體中文', modeHint = '', customApi) => request(
    '/naturalize-prompt',
    {
      method: 'POST',
      body: JSON.stringify({
        prompt_text: promptText,
        prompt_language: promptLanguage,
        mode_hint: modeHint,
        custom_api: customApi
      })
    },
    LLM_TIMEOUT_MS
  ),
  generateFinalPrompt: (id, customApi) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return request(
      '/generate-final-prompt',
      {
        method: 'POST',
        body: JSON.stringify({
          session_id: id,
          custom_api: customApi
        })
      },
      LLM_TIMEOUT_MS
    )
  },
  generatePdf: (id) => request(
    '/generate-pdf',
    {
      method: 'POST',
      body: JSON.stringify({ session_id: id })
    }
  ),
  downloadPdf: (id) => {
    if (!id || typeof id !== 'string') return Promise.reject(new Error('sessionId 缺失'))
    return requestBlob(`/download-pdf/${id}`)
  }
}
