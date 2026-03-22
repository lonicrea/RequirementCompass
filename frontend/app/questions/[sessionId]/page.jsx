'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { App, Card, Checkbox, Input, Button, Space, Typography, Modal, Radio } from 'antd'
import { api, decodeApiBaseFromQuery, saveApiBase } from '../../../lib/api'
import { removeProject } from '../../../lib/storage'

const { Title, Paragraph } = Typography

export default function QuestionsPage() {
  const { message } = App.useApp()
  const { sessionId } = useParams()
  const sessionIdText = String(sessionId || '')
  const search = useSearchParams()
  const router = useRouter()
  const [questions, setQuestions] = useState([])
  const [answers, setAnswers] = useState({})
  const [otherAnswers, setOtherAnswers] = useState({})
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [appending, setAppending] = useState(false)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const OTHER_OPTION = '其他'
  const isOtherOption = (value) => {
    const text = String(value || '').trim().toLowerCase()
    if (!text) return false
    return text.includes('other') || text.includes('其他') || text.includes('其它')
  }

  const normalizeChoiceOptions = (options) => {
    const list = Array.isArray(options) ? options : []
    const seen = new Set()
    const normalized = []
    for (const item of list) {
      let value = String(item || '').trim()
      if (!value) continue
      if (isOtherOption(value)) value = OTHER_OPTION
      if (seen.has(value)) continue
      seen.add(value)
      normalized.push(value)
    }
    if (!seen.has(OTHER_OPTION)) normalized.push(OTHER_OPTION)
    return normalized
  }

  const hasSelectedOther = (selected) => {
    if (!Array.isArray(selected)) return false
    return selected.some((value) => isOtherOption(value))
  }

  const isSingleChoiceQuestion = (question) => {
    if (String(question?.type || '') !== 'choice') return false
    const text = String(question?.text || '').toLowerCase()
    const singleTokens = [
      '生影片模型',
      '生圖模型',
      '編程模型',
      '對話模型',
      '使用哪個 ai 編程模型或助手',
      '最終提示詞使用什麼語言',
      '提示詞使用什麼語言'
    ]
    return singleTokens.some((token) => text.includes(token))
  }

  const isAssessmentQuestion = (question) => {
    const text = String(question?.text || '')
    return text.includes('階段 1｜初步需求分析')
  }

  const parseAssessmentQuestion = (text) => {
    const normalized = String(text || '')
      .replace(/\r/g, '\n')
      .replace(/^\s*\d+\.\s*/, '')
      .replace(/\s+-\s+/g, '\n- ')
      .trim()

    const lines = normalized
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)

    const title = lines.find((line) => line.includes('階段 1｜初步需求分析')) || '階段 1｜初步需求分析'
    const items = []
    let confirmText = '請先確認是否正確；若不正確，請直接修正。'

    const kvPattern = /[-•]\s*([^：:\n]+)[：:]\s*([\s\S]*?)(?=(?:\n[-•]\s*[^：:\n]+[：:])|(?:\s*請先確認是否正確)|$)/g
    let match = kvPattern.exec(normalized)
    while (match) {
      items.push({
        label: String(match[1] || '').trim(),
        value: String(match[2] || '').trim(),
      })
      match = kvPattern.exec(normalized)
    }

    if (!items.length) {
      lines.forEach((line) => {
        if (!line.startsWith('- ')) return
        const body = line.replace(/^- /, '').trim()
        const sep = body.indexOf('：')
        if (sep < 0) {
          items.push({ label: '說明', value: body })
          return
        }
        items.push({
          label: body.slice(0, sep),
          value: body.slice(sep + 1).trim(),
        })
      })
    }

    const foundConfirm = normalized.match(/請先確認是否正確[\s\S]*/)
    if (foundConfirm && foundConfirm[0]) confirmText = foundConfirm[0].trim()
    return { title, items, confirmText }
  }

  useEffect(() => {
    if (!sessionIdText) {
      message.error('缺少會話ID，已返回首頁')
      router.push('/')
      return
    }
    // 允許用 query 參數覆蓋後端位址
    const encodedApi = search.get('api')
    if (encodedApi) {
      const decoded = decodeApiBaseFromQuery(encodedApi)
      if (decoded) {
        saveApiBase(decoded)
        message.success('已切換後端地址')
      } else {
        message.error('後端地址解析失敗')
      }
      router.replace(`/questions/${sessionIdText}`)
    }
  }, [search, router, sessionIdText])

  useEffect(() => {
    const load = async () => {
      try {
        const fetchSessionWithRetry = async (maxAttempts = 2) => {
          let lastErr = null
          for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
            try {
              return await api.getSession(sessionIdText)
            } catch (err) {
              lastErr = err
              if (attempt < maxAttempts) {
                await new Promise((resolve) => setTimeout(resolve, 800))
              }
            }
          }
          throw lastErr
        }

        const res = await fetchSessionWithRetry(2)
        const qs = res.data.questions || []
        if (!qs.length) {
          message.warning('本次會話沒有問題，已返回首頁請重新開始')
          router.push('/')
          return
        }
        setQuestions(qs)
        const init = {}
        qs.forEach((q) => {
          init[q.id] = q.type === 'choice' ? [] : ''
        })
        setAnswers(init)
        setOtherAnswers({})
      } catch (err) {
        console.error(err)
        removeProject(sessionIdText)
        const backendError = err?.response?.data?.error
        if (backendError) {
          message.error(`載入問題失敗：${backendError}`)
        } else if (err?.code === 'ECONNABORTED') {
          message.error('載入問題逾時，將返回首頁')
        } else {
          message.error('載入問題失敗，將返回首頁')
        }
        router.push('/')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [sessionIdText, router])

  const submit = async () => {
    const missingSingle = questions.find((q) => {
      if (!isSingleChoiceQuestion(q)) return false
      const picked = answers[q.id]
      return !Array.isArray(picked) || picked.length === 0
    })
    if (missingSingle) {
      message.error(`請先回答：${missingSingle.text}`)
      return
    }

    setSubmitting(true)
    try {
      const payload = questions.map((q) => {
        const a = answers[q.id]
        if (q.type === 'choice') {
          const selected = Array.isArray(a) ? [...a] : []
          if (hasSelectedOther(selected)) {
            const detail = (otherAnswers[q.id] || '').trim()
            const withoutOther = selected.filter((v) => !isOtherOption(v))
            if (detail) withoutOther.push(`${OTHER_OPTION}: ${detail}`)
            return { answer: withoutOther.join(', ') }
          }
          return { answer: selected.join(', ') }
        }
        return { answer: a || '' }
      })
      const res = await api.submitAnswers(sessionIdText, payload)
      const nextId = res?.data?.session_id || sessionIdText
      router.push(`/results/${nextId}`)
    } catch (err) {
      console.error(err)
      message.error('提交失敗')
    } finally {
      setSubmitting(false)
    }
  }

  const appendQuestions = async () => {
    setAppending(true)
    try {
      const previousCount = questions.length
      const res = await api.appendQuestions(
        sessionIdText,
        '請新增一頁問題，避免與已有問題重複，並繼續深挖需求。'
      )
      const updated = res?.data?.questions || []
      if (!updated.length || updated.length <= previousCount) {
        message.warning('沒有生成新問題，請重試')
        return
      }
      setQuestions(updated)
      setAnswers((prev) => {
        const next = { ...prev }
        updated.forEach((q) => {
          if (next[q.id] === undefined) next[q.id] = q.type === 'choice' ? [] : ''
        })
        return next
      })
      setOtherAnswers((prev) => {
        const next = { ...prev }
        updated.forEach((q) => {
          if (next[q.id] === undefined) next[q.id] = ''
        })
        return next
      })
      message.success('已增加新問題')
    } catch (err) {
      console.error(err)
      message.error('增加問題失敗')
    } finally {
      setAppending(false)
    }
  }

  const submitFeedback = async () => {
    if (!feedback.trim()) {
      message.warning('請填寫補充資訊')
      return
    }
    setFeedbackLoading(true)
    try {
      const res = await api.continueWithFeedback(sessionIdText, feedback)
      setQuestions(res.data.questions)
      const init = {}
      res.data.questions.forEach((q) => {
        init[q.id] = q.type === 'choice' ? [] : ''
      })
      setAnswers(init)
      setOtherAnswers({})
      setFeedback('')
      setFeedbackOpen(false)
      message.success('已生成新問題')
    } catch (err) {
      console.error(err)
      message.error('提交失敗')
    } finally {
      setFeedbackLoading(false)
    }
  }

  const renderQuestionInput = (q, index) => {
    if (isAssessmentQuestion(q)) {
      const assessment = parseAssessmentQuestion(q.text)
      return (
        <>
          <Title level={5}>{index + 1}. {assessment.title}</Title>
          <div className="assessment-box">
            {assessment.items.map((item, itemIdx) => (
              <div className="assessment-row" key={`${q.id}-assessment-${itemIdx}`}>
                <span className="assessment-label">{item.label}</span>
                <span className="assessment-value">{item.value}</span>
              </div>
            ))}
            <div className="assessment-tip">{assessment.confirmText}</div>
          </div>
          <Input.TextArea
            rows={3}
            value={answers[q.id]}
            placeholder="若上面判斷不正確，請直接修正。若正確可留空。"
            onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
          />
        </>
      )
    }

    const title = <Title level={5}>{index + 1}. {q.text}</Title>

    if (q.type === 'choice') {
      return (
        <>
          {title}
          {isSingleChoiceQuestion(q) ? (
            <Radio.Group
              options={normalizeChoiceOptions(q.options).map((o) => ({ label: o, value: o }))}
              value={Array.isArray(answers[q.id]) ? answers[q.id][0] : ''}
              onChange={(e) => {
                const picked = [e.target.value]
                setAnswers({ ...answers, [q.id]: picked })
                if (!hasSelectedOther(picked)) {
                  setOtherAnswers({ ...otherAnswers, [q.id]: '' })
                }
              }}
            />
          ) : (
            <Checkbox.Group
              options={normalizeChoiceOptions(q.options).map((o) => ({ label: o, value: o }))}
              value={answers[q.id]}
              onChange={(vals) => {
                const picked = Array.isArray(vals) ? vals : []
                setAnswers({ ...answers, [q.id]: picked })
                if (!hasSelectedOther(picked)) {
                  setOtherAnswers({ ...otherAnswers, [q.id]: '' })
                }
              }}
            />
          )}
          {hasSelectedOther(answers[q.id]) ? (
            <Input
              style={{ marginTop: 12, maxWidth: 420 }}
              placeholder="請填寫「其他」內容"
              value={otherAnswers[q.id] || ''}
              onChange={(e) => setOtherAnswers({ ...otherAnswers, [q.id]: e.target.value })}
            />
          ) : null}
        </>
      )
    }

    if (q.type === 'narrative') {
      return (
        <>
          {title}
          <Input.TextArea
            rows={4}
            value={answers[q.id]}
            onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
          />
        </>
      )
    }

    return (
      <>
        {title}
        <Input
          value={answers[q.id]}
          onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
        />
      </>
    )
  }

  if (loading) return <div className="container">載入中...</div>

  return (
    <div className="container">
      <Card className="card landing-card form-card">
        <Title level={3}>需求澄清問答</Title>
        <Paragraph>請回答以下問題，幫助我們更好理解你的需求。</Paragraph>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          {questions.map((q, idx) => (
            <div key={q.id}>
              {renderQuestionInput(q, idx)}
            </div>
          ))}
          <div className="actions">
            <Button type="primary" loading={submitting} onClick={submit} disabled={!questions.length}>提交答案</Button>
            <Button loading={appending} onClick={appendQuestions}>增加問題</Button>
            <Button onClick={() => setFeedbackOpen(true)}>補充資訊</Button>
            <Button onClick={() => router.push('/')}>回到首頁</Button>
          </div>
        </Space>
      </Card>

      <Modal
        title="補充資訊"
        open={feedbackOpen}
        onOk={submitFeedback}
        onCancel={() => setFeedbackOpen(false)}
        confirmLoading={feedbackLoading}
      >
        <Input.TextArea rows={5} value={feedback} onChange={(e) => setFeedback(e.target.value)} />
      </Modal>
    </div>
  )
}
