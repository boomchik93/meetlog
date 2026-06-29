import React, { useEffect, useRef, useState } from 'react'
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'

const RANGES = [
  { label: '15 мин', s: 900 },
  { label: '1 час', s: 3600 },
  { label: '6 часов', s: 21600 },
  { label: '24 часа', s: 86400 },
]

const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString('ru-RU')
const fmtDate = (ts) => new Date(ts * 1000).toLocaleString('ru-RU')
const gb = (b) => (b / 1024 ** 3).toFixed(1) + ' ГБ'
const secs = (s) => s == null ? '—' : (s >= 60 ? `${Math.floor(s / 60)}м ${Math.round(s % 60)}с` : `${Math.round(s)}с`)

export default function App() {
  const [tab, setTab] = useState('load')
  const [service, setService] = useState(null)

  useEffect(() => {
    const poll = () => fetch('/api/service').then(r => r.json()).then(setService).catch(() => setService({ status: 'down' }))
    poll(); const t = setInterval(poll, 10000); return () => clearInterval(t)
  }, [])

  return (
    <div style={S.app}>
      <header style={S.header}>
        <b>🎙 Транскрибатор — админ-панель</b>
        <span style={{ ...S.badge, background: service?.status === 'ok' ? '#16a34a' : '#dc2626' }}>
          сервис: {service?.status === 'ok' ? 'работает' : 'недоступен'}
        </span>
        {service?.status === 'ok' && (
          <span style={S.muted}>
            whisper: {service.whisper?.model} · LLM: {service.llm_ready ? '✓' : '✗'}
          </span>
        )}
      </header>

      <nav style={S.nav}>
        {[['load', '📊 Нагрузка'], ['tasks', '🗂 Задачи'], ['term', '💻 Терминал']].map(([k, t]) => (
          <button key={k} onClick={() => setTab(k)} style={{ ...S.tab, ...(tab === k ? S.tabOn : {}) }}>{t}</button>
        ))}
      </nav>

      <main style={S.main}>
        {tab === 'load' && <LoadView />}
        {tab === 'tasks' && <TasksView />}
        {tab === 'term' && <TermView />}
      </main>
    </div>
  )
}

function LoadView() {
  const [range, setRange] = useState(3600)
  const [data, setData] = useState([])
  const [err, setErr] = useState(null)

  useEffect(() => {
    let live = true
    const poll = () => fetch(`/api/history?seconds=${range}`).then(r => r.json()).then(d => {
      if (!live) return
      if (Array.isArray(d)) { setData(d); setErr(null) } else setErr(d.error || 'нет данных')
    }).catch(e => setErr(String(e)))
    poll(); const t = setInterval(poll, 5000); return () => { live = false; clearInterval(t) }
  }, [range])

  const chart = data.map(d => ({ t: fmtTime(d.ts), cpu: d.cpu, mem: d.mem_pct, load1: d.load1 }))
  const last = data[data.length - 1]
  const disks = last ? Object.entries(last.disks).map(([m, v]) => ({ mount: m, pct: v.pct, used: v.used, total: v.total })) : []

  return (
    <div>
      <div style={S.row}>
        {RANGES.map(r => (
          <button key={r.s} onClick={() => setRange(r.s)} style={{ ...S.chip, ...(range === r.s ? S.chipOn : {}) }}>{r.label}</button>
        ))}
        {last && <span style={S.muted}>обновлено: {fmtTime(last.ts)} · точек: {data.length}</span>}
        {err && <span style={{ color: '#dc2626' }}>⚠ {err}</span>}
      </div>

      {!data.length && !err && <p style={S.muted}>Сбор данных… (история копится на сервере)</p>}

      {last && (
        <div style={S.cards}>
          <Stat label="CPU" value={last.cpu.toFixed(0) + '%'} />
          <Stat label="RAM" value={last.mem_pct.toFixed(0) + '%'} sub={gb(last.mem_used)} />
          <Stat label="Load (1м)" value={last.load1} sub={`5м ${last.load5} · 15м ${last.load15}`} />
        </div>
      )}

      <Panel title="CPU, %">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8883" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="cpu" stroke="#2563eb" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="RAM, %">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8883" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Area type="monotone" dataKey="mem" stroke="#16a34a" fill="#16a34a55" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Load average (1 мин)">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8883" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="load1" stroke="#d97706" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Диски (% занято)">
        <ResponsiveContainer width="100%" height={60 + disks.length * 40}>
          <BarChart data={disks} layout="vertical" margin={{ left: 30 }}>
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="mount" tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v, n, p) => [`${v}% · ${gb(p.payload.used)} / ${gb(p.payload.total)}`, 'занято']} />
            <Bar dataKey="pct" fill="#7c3aed" />
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  )
}

function TasksView() {
  const [tasks, setTasks] = useState([])
  const [open, setOpen] = useState(null)

  useEffect(() => {
    const poll = () => fetch('/api/tasks?limit=200').then(r => r.json()).then(d => Array.isArray(d) && setTasks(d)).catch(() => {})
    poll(); const t = setInterval(poll, 15000); return () => clearInterval(t)
  }, [])

  const chart = [...tasks].reverse().map(t => ({ t: fmtTime(t.ts), audio: (t.audio_sec || 0) / 60, proc: (t.proc_sec || 0) / 60 }))

  return (
    <div>
      <Panel title="Время обработки vs длина аудио (мин)">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8883" />
            <XAxis dataKey="t" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip /><Legend />
            <Bar dataKey="audio" name="аудио, мин" fill="#2563eb" />
            <Bar dataKey="proc" name="обработка, мин" fill="#d97706" />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title={`Архив задач (${tasks.length})`}>
        <table style={S.table}>
          <thead><tr><th>Когда</th><th>Файл</th><th>Тема</th><th>Аудио</th><th>Обработка</th><th>Статус</th><th></th></tr></thead>
          <tbody>
            {tasks.map((t, i) => (
              <tr key={i}>
                <td>{fmtDate(t.ts)}</td>
                <td style={S.ell}>{t.filename}</td>
                <td style={S.ell}>{t.title || '—'}</td>
                <td>{secs(t.audio_sec)}</td>
                <td>{secs(t.proc_sec)}</td>
                <td>{t.status}</td>
                <td>{t.saved_to && <button style={S.link} onClick={() => openResult(t.saved_to, setOpen)}>смотреть</button>}</td>
              </tr>
            ))}
            {!tasks.length && <tr><td colSpan="7" style={S.muted}>Пока задач нет</td></tr>}
          </tbody>
        </table>
      </Panel>

      {open && <ResultModal data={open} onClose={() => setOpen(null)} />}
    </div>
  )
}

function openResult(path, setOpen) {
  fetch('/api/result?path=' + encodeURIComponent(path))
    .then(r => r.json()).then(d => setOpen({ path, data: d })).catch(() => setOpen({ path, data: { error: 'не удалось загрузить' } }))
}

function ResultModal({ data, onClose }) {
  const d = data.data || {}
  const s = d.summary || {}
  const download = () => {
    const blob = new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = (data.path.split('/').pop()) || 'result.json'; a.click(); URL.revokeObjectURL(a.href)
  }
  return (
    <div style={S.overlay} onClick={onClose}>
      <div style={S.modal} onClick={e => e.stopPropagation()}>
        <div style={S.row}>
          <button style={S.btn} onClick={download}>⬇ Скачать JSON</button>
          <button style={S.btnGhost} onClick={onClose}>закрыть</button>
        </div>
        <h3>{s.title || '(без темы)'}</h3>
        <p>{s.summary}</p>
        {!!(s.topics || []).length && <h4>Темы</h4>}
        <ul>{(s.topics || []).map((t, i) => <li key={i}><b>{t.title}</b> [{t.category}]<ul>{(t.points || []).map((p, j) => <li key={j}>{p}</li>)}</ul></li>)}</ul>
        {!!(s.decisions || []).length && <h4>Решения</h4>}
        <ul>{(s.decisions || []).map((x, i) => <li key={i}>{x.decision} — <i>{x.responsible}</i></li>)}</ul>
        <details><summary>Сырой JSON</summary><pre style={S.pre}>{JSON.stringify(d, null, 2)}</pre></details>
      </div>
    </div>
  )
}

function TermView() {
  const ref = useRef(null)
  useEffect(() => {
    const term = new Terminal({ fontSize: 13, theme: { background: '#0b0f14' }, cursorBlink: true })
    const fit = new FitAddon(); term.loadAddon(fit)
    term.open(ref.current); fit.fit()
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/terminal`)
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => { sendResize(); term.focus() }
    ws.onmessage = (e) => term.write(typeof e.data === 'string' ? e.data : new Uint8Array(e.data))
    ws.onclose = () => term.write('\r\n[соединение закрыто]\r\n')
    term.onData(d => ws.readyState === 1 && ws.send(d))
    const sendResize = () => { fit.fit(); ws.readyState === 1 && ws.send(JSON.stringify({ resize: { cols: term.cols, rows: term.rows } })) }
    const onResize = () => sendResize()
    window.addEventListener('resize', onResize)
    return () => { window.removeEventListener('resize', onResize); ws.close(); term.dispose() }
  }, [])
  return (
    <div>
      <p style={S.muted}>Интерактивный shell на сервере. Работают top, htop, и т.д.</p>
      <div ref={ref} style={S.term} />
    </div>
  )
}

const Panel = ({ title, children }) => (
  <div style={S.panel}><div style={S.panelTitle}>{title}</div>{children}</div>
)
const Stat = ({ label, value, sub }) => (
  <div style={S.statCard}><div style={S.muted}>{label}</div><div style={S.statVal}>{value}</div>{sub && <div style={S.muted}>{sub}</div>}</div>
)

const S = {
  app: { fontFamily: 'system-ui, sans-serif', color: '#e5e7eb', background: '#0b0f14', minHeight: '100vh' },
  header: { display: 'flex', gap: 14, alignItems: 'center', padding: '12px 18px', borderBottom: '1px solid #1f2937', flexWrap: 'wrap' },
  badge: { color: '#fff', padding: '3px 10px', borderRadius: 20, fontSize: 13 },
  muted: { color: '#9ca3af', fontSize: 13 },
  nav: { display: 'flex', gap: 6, padding: '10px 18px' },
  tab: { background: '#111827', color: '#e5e7eb', border: '1px solid #1f2937', borderRadius: 8, padding: '8px 14px', cursor: 'pointer' },
  tabOn: { background: '#2563eb', borderColor: '#2563eb' },
  main: { padding: '0 18px 40px' },
  row: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', margin: '10px 0' },
  chip: { background: '#111827', color: '#e5e7eb', border: '1px solid #1f2937', borderRadius: 20, padding: '4px 12px', cursor: 'pointer', fontSize: 13 },
  chipOn: { background: '#2563eb', borderColor: '#2563eb' },
  cards: { display: 'flex', gap: 12, flexWrap: 'wrap', margin: '6px 0 16px' },
  statCard: { background: '#111827', border: '1px solid #1f2937', borderRadius: 10, padding: '12px 18px', minWidth: 120 },
  statVal: { fontSize: 26, fontWeight: 700 },
  panel: { background: '#0f1620', border: '1px solid #1f2937', borderRadius: 10, padding: 14, marginBottom: 16 },
  panelTitle: { fontSize: 13, color: '#9ca3af', marginBottom: 8 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  ell: { maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  link: { background: 'none', border: 'none', color: '#60a5fa', cursor: 'pointer', textDecoration: 'underline' },
  term: { height: '70vh', background: '#0b0f14', padding: 6, borderRadius: 8, border: '1px solid #1f2937' },
  overlay: { position: 'fixed', inset: 0, background: '#000a', display: 'flex', justifyContent: 'center', alignItems: 'flex-start', padding: 30, overflow: 'auto' },
  modal: { background: '#0f1620', border: '1px solid #1f2937', borderRadius: 12, padding: 20, maxWidth: 760, width: '100%' },
  pre: { background: '#0b0f14', padding: 10, borderRadius: 8, overflow: 'auto', maxHeight: 360, fontSize: 12 },
  btn: { background: '#2563eb', color: '#fff', border: 0, borderRadius: 8, padding: '8px 14px', cursor: 'pointer' },
  btnGhost: { background: 'none', color: '#9ca3af', border: '1px solid #1f2937', borderRadius: 8, padding: '8px 14px', cursor: 'pointer' },
}

Object.assign(S, {})
// подсветка ячеек таблицы
const css = document.createElement('style')
css.textContent = 'th,td{padding:6px 8px;border-bottom:1px solid #1f2937;text-align:left}'
document.head.appendChild(css)
