'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { App, Button, Card, Input, Radio, Select, Space, Typography } from 'antd'
import { api } from '../../lib/api'
import { upsertProject } from '../../lib/storage'

const { Title, Paragraph } = Typography

const AI_TYPE_OPTIONS = [
  '對話類：聊天、問答、客服、助理、寫作、改寫、摘要、翻譯',
  '編程類：寫程式、除錯、重構、測試生成',
  '生圖類：文字生圖、修圖、風格轉換',
  '影片類：文字生影片、影片剪輯、補幀',
  '音樂類：作曲、配樂、生成人聲與伴奏'
]

export default function StartPage() {
  const { message } = App.useApp()
  const [idea, setIdea] = useState('')
  const [selectedAiType, setSelectedAiType] = useState('')
  const [userIdentity, setUserIdentity] = useState('一般使用者')
  const [userIdentityOther, setUserIdentityOther] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const start = async () => {
    const normalizedIdea = idea.trim()
    if (!normalizedIdea) {
      message.error('請先輸入初步想法')
      return
    }
    if (!selectedAiType) {
      message.error('請先選擇一個 AI 協助類型')
      return
    }
    let identity = userIdentity.trim()
    if (userIdentity === '其他') {
      identity = userIdentityOther.trim() || '其他'
    }
    const scopedIdea = `${normalizedIdea}\n\n[期望能力類型]\n- ${selectedAiType}`
    setLoading(true)
    try {
      const res = await api.generateQuestions(scopedIdea, {
        user_identity: identity || '未提供',
        language_region: '未提供',
        existing_resources: '暫無'
      })
      const { session_id, questions } = res.data
      if (!questions || !questions.length) {
        message.error('生成的問題為空，請重試')
        return
      }
      upsertProject({ id: session_id, idea: normalizedIdea, lastVisited: new Date().toISOString() })
      router.push(`/questions/${session_id}`)
    } catch (err) {
      console.error('generateQuestions error', err?.response?.data || err)
      const backendError = err?.response?.data?.error
      message.error(backendError ? `生成問題失敗：${backendError}` : '生成問題失敗')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <Card className="card landing-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <Title level={3} style={{ marginBottom: 8 }}>開始需求對齊</Title>
          <Button onClick={() => router.push('/')}>回到首頁</Button>
        </div>

        <Paragraph>輸入你的初步想法：</Paragraph>
        <Input.TextArea
          rows={5}
          value={idea}
          onChange={(e) => setIdea(e.target.value)}
          placeholder="例如：我想做一個線上學習平臺..."
          maxLength={1000}
          showCount
        />
        <Paragraph style={{ marginTop: 16 }}>希望 AI 協助的類型（必選，單選）：</Paragraph>
        <Radio.Group
          value={selectedAiType}
          onChange={(e) => setSelectedAiType(e.target.value)}
          style={{ display: 'grid', gap: 8 }}
        >
          {AI_TYPE_OPTIONS.map((item) => (
            <Radio key={item} value={item}>
              {item}
            </Radio>
          ))}
        </Radio.Group>
        <Paragraph style={{ marginTop: 16 }}>你的身份（選填）：</Paragraph>
        <Select
          style={{ width: '100%' }}
          value={userIdentity || undefined}
          onChange={(value) => {
            setUserIdentity(value)
            setUserIdentityOther('')
          }}
          placeholder="請選擇身份"
          options={[
            { value: '一般使用者', label: '一般使用者（預設）' },
            { value: '學生', label: '學生' },
            { value: '老師', label: '老師' },
            { value: '產品經理', label: '產品經理' },
            { value: '創業者', label: '創業者' },
            { value: '開發者', label: '開發者' },
            { value: '其他', label: '其他' }
          ]}
        />
        {userIdentity === '其他' ? (
          <Input
            style={{ marginTop: 12 }}
            value={userIdentityOther}
            onChange={(e) => setUserIdentityOther(e.target.value)}
            placeholder="請填寫你的身份"
            maxLength={60}
          />
        ) : null}
        <div style={{ marginTop: 18 }}>
          <Space size={12}>
            <Button type="primary" size="large" loading={loading} onClick={start}>
              開始需求對齊
            </Button>
          </Space>
        </div>
      </Card>
    </div>
  )
}
