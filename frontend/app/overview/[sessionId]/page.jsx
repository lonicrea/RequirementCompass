'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { Card, Typography, Timeline, Collapse, Button, message } from 'antd'
import { api, decodeApiBaseFromQuery, saveApiBase } from '../../../lib/api'
import { marked } from 'marked'

const { Title, Paragraph } = Typography

export default function OverviewPage() {
  const { sessionId } = useParams()
  const sessionIdText = String(sessionId || '')
  const search = useSearchParams()
  const router = useRouter()
  const [idea, setIdea] = useState('')
  const [rounds, setRounds] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const encodedApi = search.get('api')
    if (encodedApi) {
      const decoded = decodeApiBaseFromQuery(encodedApi)
      if (decoded) {
        saveApiBase(decoded)
        message.success('已切換後端地址')
      } else {
        message.error('後端地址解析失敗')
      }
      router.replace(`/overview/${sessionIdText}`)
    }
  }, [search, router, sessionIdText])

  useEffect(() => {
    const load = async () => {
      try {
        const [sessionRes, roundsRes] = await Promise.all([
          api.getSession(sessionIdText),
          api.getRounds(sessionIdText)
        ])
        setIdea(sessionRes.data.idea)
        setRounds(roundsRes.data.rounds || [])
      } catch (err) {
        console.error(err)
        message.error('載入失敗')
      } finally {
        setLoading(false)
      }
    }
    if (sessionIdText) load()
  }, [sessionIdText])

  const downloadFullProcess = () => {
    const lines = ['# 專案需求對齊完整過程', '', `- 專案 ID: ${sessionIdText}`, '', '## 原始想法', '', idea || '']
    rounds.forEach((round) => {
      lines.push('', `## 第 ${round.round_number} 輪`, '')
      ;(round.questions || []).forEach((q, idx) => {
        const answer = (round.answers || [])[idx]?.answer || ''
        lines.push(`### Q${idx + 1}: ${q.text || ''}`, '', `A${idx + 1}: ${answer}`, '')
      })
      if (round.report) lines.push('### 階段報告', '', round.report, '')
    })

    const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `project_overview_${sessionIdText.slice(0, 8)}.md`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  }

  return (
    <div className="container">
      <Card className="card landing-card form-card" loading={loading}>
        <Title level={3}>專案總覽</Title>
        <Paragraph><strong>專案ID：</strong>{sessionId}</Paragraph>
        <Paragraph><strong>原始想法：</strong>{idea}</Paragraph>
        <Timeline items={rounds.map((r) => ({
          color: 'blue',
          children: (
            <div>
              <Title level={5}>第 {r.round_number} 輪</Title>
              <Collapse
                items={[{
                  key: 'qa',
                  label: '問答內容',
                  children: (
                    <div>
                      {(r.questions || []).map((q, idx) => (
                        <div key={idx} style={{ marginBottom: 8 }}>
                          <div><strong>Q{idx+1}:</strong> {q.text}</div>
                          <div style={{ color: '#2f7a1f' }}><strong>A{idx+1}:</strong> {(r.answers[idx] && r.answers[idx].answer) || ''}</div>
                        </div>
                      ))}
                    </div>
                  )
                }, {
                  key: 'report',
                  label: '階段報告',
                  children: <div dangerouslySetInnerHTML={{ __html: marked.parse(r.report || '無') }} />
                }]}
              />
            </div>
          )
        }))} />
        <div className="actions" style={{ marginTop: 16 }}>
          <Button type="primary" onClick={downloadFullProcess}>下載完整對齊過程</Button>
          <Button onClick={() => router.push(`/results/${sessionIdText}`)}>返回報告</Button>
          <Button onClick={() => router.push('/')}>回到首頁</Button>
        </div>
      </Card>
    </div>
  )
}
