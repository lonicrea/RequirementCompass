import 'antd/dist/reset.css'
import './globals.css'
import { ConfigProvider, App as AntApp } from 'antd'

const PRIMARY = '#4f51ab'
const PRIMARY_HOVER = '#6265be'
const PRIMARY_ACTIVE = '#424595'

export const metadata = {
  title: '需求羅盤',
  description: '需求羅盤前端介面'
}

export default function RootLayout({ children }) {
  return (
    <html lang="zh-Hant">
      <body>
        <ConfigProvider
          theme={{
            token: {
              colorPrimary: PRIMARY,
              colorInfo: PRIMARY,
              colorPrimaryHover: PRIMARY_HOVER,
              colorPrimaryActive: PRIMARY_ACTIVE
            }
          }}
        >
          <AntApp>
            {children}
          </AntApp>
        </ConfigProvider>
      </body>
    </html>
  )
}
