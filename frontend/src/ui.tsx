import { useEffect, useId, useState, type ReactNode } from 'react'
import { clock, parseTags } from './lib.ts'

export function Spinner() {
  return <span className="spinner" role="presentation" />
}

/** A running job with a clock, so a two-minute wait does not look like a freeze. */
export function Working({ label, hint }: { label: string; hint?: string }) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="working" role="status" aria-live="polite">
      <Spinner />
      <div>
        <strong>{label}</strong> <span className="muted">{clock(seconds)}</span>
        {hint && <div className="muted small">{hint}</div>}
      </div>
    </div>
  )
}

export function ErrorBanner({ message, onRetry, retryLabel = 'Try again' }: { message: string | null; onRetry?: () => void; retryLabel?: string }) {
  if (!message) return null
  return (
    <div className="banner error" role="alert">
      <span>{message}</span>
      {onRetry && <button type="button" className="btn small" onClick={onRetry}>{retryLabel}</button>}
    </div>
  )
}

export function Notice({ tone = 'info', children }: { tone?: 'info' | 'warn' | 'good'; children: ReactNode }) {
  return <div className={`banner ${tone}`}>{children}</div>
}

export function Badge({ tone, children }: { tone: 'good' | 'warn' | 'bad' | 'neutral'; children: ReactNode }) {
  return <span className={`badge ${tone}`}>{children}</span>
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: (id: string) => ReactNode }) {
  const id = useId()
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children(id)}
      {hint && <div className="muted small">{hint}</div>}
    </div>
  )
}

export function TextInput({ label, value, onChange, hint, type = 'text', maxLength }: {
  label: string; value: string; onChange: (v: string) => void; hint?: string; type?: string; maxLength?: number
}) {
  return (
    <Field label={label} hint={hint}>
      {(id) => <input id={id} type={type} value={value} maxLength={maxLength} onChange={(e) => onChange(e.target.value)} />}
    </Field>
  )
}

export function TextArea({ label, value, onChange, hint, rows = 4, maxLength }: {
  label: string; value: string; onChange: (v: string) => void; hint?: string; rows?: number; maxLength?: number
}) {
  return (
    <Field label={label} hint={maxLength ? `${hint ? hint + ' · ' : ''}${value.length}/${maxLength}` : hint}>
      {(id) => <textarea id={id} rows={rows} value={value} maxLength={maxLength} onChange={(e) => onChange(e.target.value)} />}
    </Field>
  )
}

/** Comma-separated values in one box (skills, technologies). Edits are kept as typed and cleaned when committed. */
export function TagsInput({ label, value, onChange, hint }: { label: string; value: string[]; onChange: (v: string[]) => void; hint?: string }) {
  const joined = value.join(', ')
  const [text, setText] = useState(joined)
  useEffect(() => {
    // follow outside changes (e.g. a regenerated draft) but never overwrite what the user is typing
    if (parseTags(text).join('|') !== value.join('|')) setText(joined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [joined])
  return (
    <Field label={label} hint={hint ?? 'Separate with commas'}>
      {(id) => (
        <input id={id} value={text} onChange={(e) => { setText(e.target.value); onChange(parseTags(e.target.value)) }}
          onBlur={() => setText(parseTags(text).join(', '))} />
      )}
    </Field>
  )
}

/** A list of free-text entries (experience, projects, achievements): one box each, add and remove. */
export function EntryList({ label, items, onChange, rows = 5, addLabel = 'Add another', hint }: {
  label: string; items: string[]; onChange: (items: string[]) => void; rows?: number; addLabel?: string; hint?: string
}) {
  return (
    <fieldset className="entries">
      <legend>{label}</legend>
      {hint && <div className="muted small">{hint}</div>}
      {items.map((item, i) => (
        <div className="entry" key={i}>
          <textarea aria-label={`${label} ${i + 1}`} rows={rows} value={item}
            onChange={(e) => onChange(items.map((x, j) => (j === i ? e.target.value : x)))} />
          <button type="button" className="btn small ghost" onClick={() => onChange(items.filter((_, j) => j !== i))}
            aria-label={`Remove ${label} ${i + 1}`}>Remove</button>
        </div>
      ))}
      <button type="button" className="btn small" onClick={() => onChange([...items, ''])}>{addLabel}</button>
    </fieldset>
  )
}

export function Collapsible({ title, count, children, open = false }: { title: string; count?: number; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="collapsible">
      <summary>{title}{count !== undefined && <span className="muted"> ({count})</span>}</summary>
      {children}
    </details>
  )
}

export function StepHeader({ title, intro }: { title: string; intro?: string }) {
  return (
    <header className="step-header">
      <h2>{title}</h2>
      {intro && <p className="muted">{intro}</p>}
    </header>
  )
}
