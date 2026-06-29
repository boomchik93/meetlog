import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' — чтобы ассеты грузились, когда бэкенд отдаёт dist из корня
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    // для разработки: vite dev проксирует api/ws на бэкенд
    proxy: {
      '/api': 'http://localhost:8765',
      '/ws': { target: 'ws://localhost:8765', ws: true },
    },
  },
})
