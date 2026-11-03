import { useEffect, useRef, useState } from 'react'
import { ChooseTemplateError, createCompany, createOutreach, createRecipient, editEmail, emailAttachment, rejectEmail, sendEmail } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { errorText, looksLikeEmail } from '../lib.ts'
import type { EmailRecord, PdfResult } from '../types.ts'
import { Badge, ErrorBanner, Field, Notice, StepHeader, TextArea, TextInput, Working } from '../ui.tsx'

const SUGGESTED = ['Why should you choose me?', 'Why are you a good fit for this role?']

export function EmailStep({ onRestart }: { onRestart: () => void }) {
  const { flow, update } = useFlow()
  const [company, setCompany] = useState(flow.job.company)
  const [website, setWebsite] = useState('')
  const [recipientName, setRecipientName] = useState('')
  const [recipientEmail, setRecipientEmail] = useState('')
  const [role, setRole] = useState('')
  const [note, setNote] = useState('')
  const [questions, setQuestions] = useState<string[]>([])
  const [email, setEmail] = useState<EmailRecord | null>(null)
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [pdf, setPdf] = useState<PdfResult | null>(null)
  const [busy, setBusy] = useState<null | 'writing' | 'saving' | 'preview' | 'sending'>(null)
  const [error, setError] = useState<string | null>(null)
  const urlRef = useRef<string | null>(null)
  const draft = flow.draft
  useEffect(() => () => { if (urlRef.current) URL.revokeObjectURL(urlRef.current) }, [])

  if (!flow.profile || !draft || draft.status !== 'approved' || !flow.template) {
    return <section><StepHeader title="Write to the founder" /><Notice tone="warn">Approve your tailored resume and choose a design first.</Notice></section>
  }

  const sent = email?.status === 'sent'
  const problems: string[] = []
  if (!company.trim()) problems.push('Enter the company name.')
  if (!looksLikeEmail(recipientEmail)) problems.push('Enter the recipient\'s e-mail address.')
  if (note.length > 240) problems.push('The personal note can be at most 240 characters.')
  const badQuestion = questions.find((q) => q.trim().length < 3 || q.trim().length > 200)

  const toggleQuestion = (q: string) => setQuestions((qs) => (qs.includes(q) ? qs.filter((x) => x !== q) : qs.length < 4 ? [...qs, q] : qs))

  const write = async () => {
    setBusy('writing'); setError(null)
    try {
      const c = await createCompany(company.trim(), website.trim() || null)
      const r = await createRecipient(recipientName.trim() || null, recipientEmail.trim(), role.trim() || null)
      const created = await createOutreach({
        profile_id: flow.profile!.id, company_id: c.id, recipient_id: r.id, resume_draft_id: draft.id,
        questions: questions.map((q) => q.trim()).filter(Boolean), personal_note: note.trim() || null,
      })
      if (created.status === 'failed') throw new Error(created.error || 'The e-mail could not be written. Try again.')
      setEmail(created); setSubject(created.subject ?? ''); setBody(created.body ?? '')
      update({ emailId: created.id })
    } catch (err) { setError(errorText(err)) } finally { setBusy(null) }
  }

  const dirty = !!email && (subject !== (email.subject ?? '') || body !== (email.body ?? ''))
  const save = async () => {
    if (!email) return email
    if (!dirty) return email
    setBusy('saving'); setError(null)
    try {
      const saved = await editEmail(email.id, { subject, body })
      setEmail(saved); setSubject(saved.subject ?? ''); setBody(saved.body ?? '')
      return saved
    } catch (err) { setError(errorText(err)); return null } finally { setBusy(null) }
  }

  const showAttachment = async () => {
    if (!email) return
    setBusy('preview'); setError(null)
    try {
      const result = await emailAttachment(email.id, flow.template)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = result.url
      setPdf(result)
    } catch (err) {
      setError(err instanceof ChooseTemplateError ? `${err.message} Go back and choose another design.` : errorText(err))
    } finally { setBusy(null) }
  }

  const send = async () => {
    if (!email) return
    if (!window.confirm(`Send this e-mail to ${recipientEmail.trim()} now, with your resume attached? This cannot be undone.`)) return
    const saved = await save()
    if (!saved) return
    setBusy('sending'); setError(null)
    try {
      const done = await sendEmail(email.id, flow.template)
      if (done.status === 'failed') throw new Error(done.error || 'The e-mail could not be sent.')
      setEmail(done)
    } catch (err) { setError(errorText(err)) } finally { setBusy(null) }
  }

  const discard = async () => {
    if (!email || !window.confirm('Discard this e-mail?')) return
    try { await rejectEmail(email.id) } catch { /* it may already be gone; either way start fresh */ }
    setEmail(null); setPdf(null); update({ emailId: null })
  }

  if (sent) {
    return (
      <section>
        <StepHeader title="Sent" />
        <Notice tone="good">Your e-mail was sent to <strong>{recipientEmail}</strong> with your tailored resume attached.</Notice>
        <div className="actions"><button className="btn primary" onClick={onRestart}>Apply to another job</button></div>
      </section>
    )
  }

  if (!email) {
    return (
      <section>
        <StepHeader title="Write to the founder" intro="We write a short e-mail from your real experience, with your tailored resume attached. You review it before anything is sent." />
        <div className="grid2">
          <TextInput label="Company" value={company} onChange={setCompany} />
          <TextInput label="Company website (optional)" value={website} onChange={setWebsite} />
          <TextInput label="Recipient's name (optional)" value={recipientName} onChange={setRecipientName} hint="Without a name the greeting is “Hi {company} Team,”." />
          <TextInput label="Recipient's e-mail" type="email" value={recipientEmail} onChange={setRecipientEmail} />
          <TextInput label="Their role (optional)" value={role} onChange={setRole} />
        </div>
        <TextArea label="One personal line (optional)" rows={2} maxLength={240} value={note} onChange={setNote}
          hint="Why this company? Used exactly as you write it, at the start of the intro." />
        <Field label="Questions they asked (optional)" hint="Each one gets its own short answer with proof from your resume.">
          {() => (
            <div className="chips">
              {SUGGESTED.map((q) => (
                <button key={q} type="button" className={`chip ${questions.includes(q) ? 'on' : ''}`} onClick={() => toggleQuestion(q)} aria-pressed={questions.includes(q)}>{q}</button>
              ))}
            </div>
          )}
        </Field>
        {questions.filter((q) => !SUGGESTED.includes(q)).map((q) => <p key={q} className="small">• {q}</p>)}
        <CustomQuestion onAdd={(q) => setQuestions((qs) => (qs.length < 4 && !qs.includes(q) ? [...qs, q] : qs))} />
        {problems.map((p) => <p key={p} className="error-text">{p}</p>)}
        {badQuestion && <p className="error-text">Each question must be 3 to 200 characters.</p>}
        {busy === 'writing' && <Working label="Writing your e-mail" hint="Usually under half a minute." />}
        <ErrorBanner message={error} onRetry={write} />
        <div className="actions"><button className="btn primary" disabled={busy !== null || problems.length > 0 || !!badQuestion} onClick={write}>Write my e-mail</button></div>
      </section>
    )
  }

  return (
    <section>
      <StepHeader title="Review your e-mail" intro={`To ${recipientEmail.trim()}. Edit freely: what you see here is exactly what is sent.`} />
      <TextInput label="Subject" value={subject} onChange={setSubject} maxLength={150} />
      <TextArea label="E-mail" value={body} onChange={setBody} rows={18} maxLength={8000} />
      <p className="small muted">Attached: your tailored resume ({flow.template === 'original' ? 'your own design' : `${flow.template} template`}).</p>
      <ErrorBanner message={error} />
      {busy && <Working label={busy === 'sending' ? 'Sending' : busy === 'preview' ? 'Building the attachment' : 'Saving'} />}
      {pdf && <iframe className="pdf" title="E-mail attachment preview" src={pdf.url} />}
      <div className="actions">
        <button className="btn" disabled={busy !== null} onClick={showAttachment}>{pdf ? 'Refresh attachment' : 'Preview attachment'}</button>
        <button className="btn" disabled={busy !== null || !dirty} onClick={() => void save()}>Save edits</button>
        <button className="btn primary" disabled={busy !== null || !subject.trim() || !body.trim()} onClick={send}>Send now</button>
        <button className="btn ghost" disabled={busy !== null} onClick={discard}>Discard</button>
        {dirty && <Badge tone="warn">unsaved edits</Badge>}
      </div>
    </section>
  )
}

function CustomQuestion({ onAdd }: { onAdd: (q: string) => void }) {
  const [text, setText] = useState('')
  const add = () => { const q = text.trim().replace(/\s+/g, ' '); if (q.length >= 3 && q.length <= 200) { onAdd(q); setText('') } }
  return (
    <div className="inline">
      <input aria-label="Another question" placeholder="Another question they asked" value={text} onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add() } }} />
      <button type="button" className="btn small" onClick={add} disabled={text.trim().length < 3}>Add</button>
    </div>
  )
}
