/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// The browser only ever talks to /api; Vite forwards it to the FastAPI server, so there is no CORS to
// configure and no backend address in the source. Point it elsewhere with VITE_BACKEND_URL.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backend = env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'
  // The role-fit review can take a couple of minutes on a local model, so the proxy must not give up early.
  const patient = { target: backend, changeOrigin: true, timeout: 600_000, proxyTimeout: 600_000 }
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { '/api': { ...patient, rewrite: (path: string) => path.replace(/^\/api/, '') } } },
    preview: { port: 4173, proxy: { '/api': { ...patient, rewrite: (path: string) => path.replace(/^\/api/, '') } } },
    test: { environment: 'jsdom', globals: true, setupFiles: ['./src/test-setup.ts'], css: false },
  }
})
