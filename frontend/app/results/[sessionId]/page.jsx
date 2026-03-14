'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { Card, Button, Typography, Space, message } from 'antd'
import { api, decodeApiBaseFromQuery, saveApiBase } from '../../../lib/api'
import { marked } from 'marked'

const { Title, Paragraph } = Typography

const stripOuterCodeFence = (text) => {
  const source = String(text || '')
  const trimmed = source.trim()
  if (!trimmed.startsWith('```') || !trimmed.endsWith('```')) return source
  return trimmed.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '').trim()
}

export default function ResultsPage() {
  const { sessionId } = useParams()
  const sessionIdText = String(sessionId || '')
  const search = useSearchParams()
  const router = useRouter()
  const [report, setReport] = useState('')
  const [loading, setLoading] = useState(true)
  const reportText = stripOuterCodeFence(report || '暫無報告').trim()
  const reportHtml = marked.parse(reportText)

  useEffect(() => {
    const encodedApi = search.get('api')
    if (!encodedApi) return
    const decoded = decodeApiBaseFromQuery(encodedApi)
    if (decoded) {
      saveApiBase(decoded)
      message.success('已切換後端地址')
    } else {
      message.error('後端地址解析失敗')
    }
    router.replace(`/results/${sessionIdText}`)
  }, [search, router, sessionIdText])

  useEffect(() => {
    const load = async () => {
      try {
        const res = await api.getSession(sessionIdText)
        const reps = res.data.reports || []
        setReport(reps[reps.length - 1] || '')
      } catch (err) {
        console.error(err)
        message.error('載入報告失敗')
      } finally {
        setLoading(false)
      }
    }
    if (sessionIdText) load()
  }, [sessionIdText])

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(reportText)
      message.success('提示詞已複製')
    } catch (err) {
      console.error(err)
      message.error('複製失敗')
    }
  }

  return (
    <div className="container">
      <Card className="card landing-card results-card" loading={loading}>
        <div className="report-hero">
          <div>
            <Title level={2} className="report-title">需求分析報告</Title>
            <Paragraph className="report-subtitle">
              已根據你的對話內容整理成可直接使用的報告，下面可直接複製、分享或繼續細化。
            </Paragraph>
          </div>
        </div>

        <div className="report-panel">
          <div className="markdown-body report-markdown" dangerouslySetInnerHTML={{ __html: reportHtml }} />
        </div>

        <Space className="actions report-actions" style={{ marginTop: 20 }}>
          <Button type="primary" onClick={copyPrompt}>複製提示詞</Button>
          <Button onClick={() => router.push(`/overview/${sessionIdText}`)}>查看總覽</Button>
          <Button onClick={() => router.push('/')}>回到首頁</Button>
        </Space>
      </Card>
    </div>
  )
}
