import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Info,
  Plus,
  Trash2,
} from 'lucide-react'
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
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <strong className="text-white font-medium">{label}</strong>
          <span className="inline-flex items-center gap-1 text-xs text-indigo-300 font-mono bg-indigo-950/60 px-2 py-0.5 rounded-md border border-indigo-800/40">
            <Clock size={12} />
            {clock(seconds)}
          </span>
        </div>
        {hint && <div className="muted small mt-1">{hint}</div>}
      </div>
    </div>
  )
}

export function ErrorBanner({
  message,
  onRetry,
  retryLabel = 'Try again',
}: {
  message: string | null
  onRetry?: () => void
  retryLabel?: string
}) {
  if (!message) return null
  return (
    <div className="banner error" role="alert">
      <div className="flex items-center gap-2.5">
        <AlertCircle size={18} className="text-rose-400 shrink-0" />
        <span className="font-medium">{message}</span>
      </div>
      {onRetry && (
        <button type="button" className="btn small border-rose-500/30 hover:border-rose-400" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
    </div>
  )
}

export function Notice({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'warn' | 'good'
  children: ReactNode
}) {
  const icon =
    tone === 'good' ? (
      <CheckCircle2 size={18} className="text-emerald-400 shrink-0 mt-0.5" />
    ) : tone === 'warn' ? (
      <AlertTriangle size={18} className="text-amber-400 shrink-0 mt-0.5" />
    ) : (
      <Info size={18} className="text-indigo-400 shrink-0 mt-0.5" />
    )
  return (
    <div className={`banner ${tone}`}>
      <div className="flex items-start gap-2.5">
        {icon}
        <div>{children}</div>
      </div>
    </div>
  )
}

export function Badge({
  tone,
  children,
}: {
  tone: 'good' | 'warn' | 'bad' | 'neutral'
  children: ReactNode
}) {
  const dotColor =
    tone === 'good'
      ? 'bg-emerald-400'
      : tone === 'warn'
      ? 'bg-amber-400'
      : tone === 'bad'
      ? 'bg-rose-400'
      : 'bg-slate-400'
  return (
    <span className={`badge ${tone}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      {children}
    </span>
  )
}

export function Field({
  label,
  hint,
  actions,
  children,
}: {
  label: string
  hint?: string
  /** Controls (badges, buttons) shown on the right of the label row. */
  actions?: ReactNode
  children: (id: string) => ReactNode
}) {
  const id = useId()
  return (
    <div className="field">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label htmlFor={id}>{label}</label>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : hint && <span className="muted small">{hint}</span>}
      </div>
      {children(id)}
      {actions && hint && <span className="muted small">{hint}</span>}
    </div>
  )
}

export function TextInput({
  label,
  value,
  onChange,
  hint,
  type = 'text',
  maxLength,
  actions,
  disabled,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  hint?: string
  type?: string
  maxLength?: number
  actions?: ReactNode
  disabled?: boolean
}) {
  return (
    <Field label={label} hint={hint} actions={actions}>
      {(id) => (
        <input
          id={id}
          type={type}
          value={value}
          maxLength={maxLength}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </Field>
  )
}

export function TextArea({
  label,
  value,
  onChange,
  hint,
  rows = 4,
  maxLength,
  autoResize = false,
  actions,
  disabled,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  hint?: string
  rows?: number
  maxLength?: number
  /** Grow and shrink with the text instead of scrolling inside the box. */
  autoResize?: boolean
  actions?: ReactNode
  disabled?: boolean
}) {
  const counter = maxLength ? `${value.length}/${maxLength}` : undefined
  const displayHint = hint && counter ? `${hint} · ${counter}` : hint || counter
  const box = useRef<HTMLTextAreaElement>(null)
  // measured before the browser paints, so the box is the right height on every keystroke and on first show
  useLayoutEffect(() => {
    const el = box.current
    if (!autoResize || !el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight + 2}px`
  }, [value, autoResize])
  return (
    <Field label={label} hint={displayHint} actions={actions}>
      {(id) => (
        <textarea
          id={id}
          ref={box}
          rows={rows}
          value={value}
          maxLength={maxLength}
          disabled={disabled}
          style={autoResize ? { overflow: 'hidden', resize: 'none' } : undefined}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </Field>
  )
}

/** Comma-separated values in one box (skills, technologies). Edits are kept as typed and cleaned when committed. */
export function TagsInput({
  label,
  value,
  onChange,
  hint,
}: {
  label: string
  value: string[]
  onChange: (v: string[]) => void
  hint?: string
}) {
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
        <div className="space-y-2">
          <input
            id={id}
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              onChange(parseTags(e.target.value))
            }}
            onBlur={() => setText(parseTags(text).join(', '))}
            placeholder="Type items separated by commas..."
          />
          {value.length > 0 && (
            <div className="chips pt-1">
              {value.map((tag, i) => (
                <span
                  key={i}
                  className="inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-md bg-indigo-950/40 border border-indigo-800/40 text-indigo-300"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </Field>
  )
}

/** A list of free-text entries (experience, projects, achievements): one box each, add and remove. */
export function EntryList({
  label,
  items,
  onChange,
  rows = 5,
  addLabel = 'Add another',
  hint,
}: {
  label: string
  items: string[]
  onChange: (items: string[]) => void
  rows?: number
  addLabel?: string
  hint?: string
}) {
  return (
    <fieldset className="entries">
      <legend>{label}</legend>
      {hint && <div className="muted small mb-3">{hint}</div>}
      <div className="space-y-3">
        {items.map((item, i) => (
          <div className="entry" key={i}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-mono font-medium text-slate-400">
                Item #{i + 1}
              </span>
              <button
                type="button"
                className="btn small ghost text-rose-400 hover:text-rose-300 hover:bg-rose-950/30"
                onClick={() => onChange(items.filter((_, j) => j !== i))}
                aria-label={`Remove ${label} ${i + 1}`}
              >
                <Trash2 size={13} className="mr-1 inline" />
                Remove
              </button>
            </div>
            <textarea
              aria-label={`${label} ${i + 1}`}
              rows={rows}
              value={item}
              onChange={(e) =>
                onChange(items.map((x, j) => (j === i ? e.target.value : x)))
              }
            />
          </div>
        ))}
      </div>
      <div className="mt-3">
        <button
          type="button"
          className="btn small"
          onClick={() => onChange([...items, ''])}
        >
          <Plus size={14} className="mr-1" />
          {addLabel}
        </button>
      </div>
    </fieldset>
  )
}

export function Collapsible({
  title,
  count,
  children,
  open = false,
}: {
  title: string
  count?: number
  children: ReactNode
  open?: boolean
}) {
  return (
    <details open={open} className="collapsible">
      <summary>
        {title}
        {count !== undefined && (
          <span className="ml-2 text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
            {count}
          </span>
        )}
      </summary>
      <div className="pt-3">{children}</div>
    </details>
  )
}

export function StepHeader({
  title,
  intro,
}: {
  title: string
  intro?: string
}) {
  return (
    <header className="step-header">
      <h2>{title}</h2>
      {intro && <p className="muted text-sm mt-1">{intro}</p>}
    </header>
  )
}
