import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  FileCheck,
  Mail,
  Paperclip,
  Plus,
  RefreshCw,
  RotateCcw,
  Send,
  Sparkles,
  Trash2,
  Undo2,
  Wand2,
} from 'lucide-react'
import {
  ChooseTemplateError,
  actionOf,
  createCompany,
  createOutreach,
  createRecipient,
  editEmail,
  emailAttachment,
  emailAttachmentInfo,
  regenerateEmailPart,
  rejectEmail,
  sendEmail,
  suggestAnswer,
  suggestPersonalLine,
  type OutreachContext,
} from '../api.ts'
import { Source } from '../components/PositioningSection.tsx'
import { useFlow } from '../flow.tsx'
import { bodyWords, errorText, extractQuestions, looksLikeEmail } from '../lib.ts'
import type { AttachmentInfo, EmailRecord, PdfResult, PersonalLineResult } from '../types.ts'
import { Badge, ErrorBanner, Notice, StepHeader, TextArea, TextInput, Working } from '../ui.tsx'

const SUGGESTED = ['Why should you choose me?', 'Why are you a good fit for this role?']
const MAX_QUESTIONS = 4
const WORDS_MIN = 120
const WORDS_MAX = 180
const ANSWER_WORDS = 45

/** One question, its answer, and who wrote the answer (the AI's latest text is `base`; anything else was edited by the user). */
interface QA {
  question: string
  answer: string
  base: string
  evidence: string[]
  variant: number
  include: boolean
  busy: boolean
  error: string | null
}

const wordsOf = (text: string) => text.split(/\s+/).filter(Boolean).length

export function EmailStep({ onRestart }: { onRestart: () => void }) {
  const { flow, update } = useFlow()
  const [company, setCompany] = useState(flow.job.company)
  const [website, setWebsite] = useState('')
  const [recipientName, setRecipientName] = useState('')
  const [recipientEmail, setRecipientEmail] = useState('')
  const [role, setRole] = useState('')

  // the personal line
  const [note, setNote] = useState('')
  const [noteBase, setNoteBase] = useState('')
  const [noteInfo, setNoteInfo] = useState<PersonalLineResult | null>(null)
  const [noteVariant, setNoteVariant] = useState(0)
  const [noteBusy, setNoteBusy] = useState(false)
  const [noteError, setNoteError] = useState<string | null>(null)

  // questions and answers
  const [qas, setQas] = useState<QA[]>([])

  // the e-mail
  const [email, setEmail] = useState<EmailRecord | null>(null)
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [subjectBase, setSubjectBase] = useState('')
  const [bodyBase, setBodyBase] = useState('')
  const [variants, setVariants] = useState({ subject: 0, body: 0 })
  const [regenerating, setRegenerating] = useState<null | 'subject' | 'body'>(null)
  const [attachment, setAttachment] = useState<AttachmentInfo | null>(null)
  const [pdf, setPdf] = useState<PdfResult | null>(null)
  const [busy, setBusy] = useState<null | 'writing' | 'saving' | 'preview' | 'sending'>(null)
  const [error, setError] = useState<string | null>(null)

  const ids = useRef<{ key: string; companyId: string; recipientId: string } | null>(null)
  const urlRef = useRef<string | null>(null)
  const draft = flow.draft
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
  }, [])

  const approved = draft?.status === 'approved'
  // The resume can be attached only when it is approved AND a design was chosen. Never the original upload.
  const canAttach = approved && !!flow.template
  const resumeDraftId = approved ? draft!.id : null

  const jobContext = { job_description: flow.job.description.trim() || null, job_title: flow.job.title.trim() || null }
  const entityKey = JSON.stringify([company.trim(), website.trim(), recipientName.trim(), recipientEmail.trim(), role.trim()])

  /** The company and recipient are saved once and reused; they are saved again only if the user changed them. */
  const ensureEntities = async (): Promise<OutreachContext> => {
    if (!ids.current || ids.current.key !== entityKey) {
      const c = await createCompany(company.trim(), website.trim() || null)
      const r = await createRecipient(recipientName.trim() || null, recipientEmail.trim(), role.trim() || null)
      ids.current = { key: entityKey, companyId: c.id, recipientId: r.id }
    }
    return { profile_id: flow.profile!.id, company_id: ids.current.companyId, recipient_id: ids.current.recipientId, resume_draft_id: resumeDraftId, ...jobContext }
  }

  if (!flow.profile) {
    return (
      <section className="space-y-4">
        <StepHeader title="Write to the founder" />
        <Notice tone="warn">Upload and confirm your resume first.</Notice>
      </section>
    )
  }

  const sent = email?.status === 'sent'
  const problems: string[] = []
  if (!company.trim()) problems.push('Enter the company name.')
  if (!looksLikeEmail(recipientEmail)) problems.push("Enter the recipient's e-mail address.")
  if (note.length > 240) problems.push('The personal line can be at most 240 characters.')
  const included = qas.filter((q) => q.include && q.answer.trim())
  const badAnswer = included.find((q) => wordsOf(q.answer) > ANSWER_WORDS)
  const ready = problems.length === 0

  // ---- the personal line ----
  const noteEdited = note.trim() !== noteBase.trim()
  const writeNote = async () => {
    if (note.trim() && noteEdited && !window.confirm('Replace your edits to the personal line with a new suggestion?')) return
    setNoteBusy(true)
    setNoteError(null)
    try {
      const variant = noteInfo ? noteVariant + 1 : 0
      const out = await suggestPersonalLine(await ensureEntities(), variant)
      setNote(out.text)
      setNoteBase(out.text)
      setNoteInfo(out)
      setNoteVariant(variant)
    } catch (err) {
      setNoteError(errorText(err))
    } finally {
      setNoteBusy(false)
    }
  }

  // ---- questions ----
  const patchQa = (question: string, patch: Partial<QA>) =>
    setQas((list) => list.map((q) => (q.question === question ? { ...q, ...patch } : q)))

  const generateAnswer = async (question: string, variant: number) => {
    patchQa(question, { busy: true, error: null })
    try {
      const out = await suggestAnswer(await ensureEntities(), question, variant)
      patchQa(question, { answer: out.answer, base: out.answer, evidence: out.evidence, variant, busy: false })
    } catch (err) {
      patchQa(question, { busy: false, error: errorText(err) })
    }
  }

  const addQuestion = (raw: string) => {
    const q = raw.trim().replace(/\s+/g, ' ')
    if (q.length < 3 || q.length > 200 || qas.length >= MAX_QUESTIONS || qas.some((x) => x.question === q)) return
    setQas((list) => [...list, { question: q, answer: '', base: '', evidence: [], variant: 0, include: false, busy: false, error: null }])
    if (ready) void generateAnswer(q, 0)
  }
  const removeQuestion = (q: string) => setQas((list) => list.filter((x) => x.question !== q))
  const regenerateAnswer = (qa: QA) => {
    if (qa.answer.trim() !== qa.base.trim() && !window.confirm('Replace your edits to this answer with a new one?')) return
    void generateAnswer(qa.question, qa.variant + 1)
  }
  const fromJob = extractQuestions(flow.job.description).filter((q) => !qas.some((x) => x.question === q))

  // ---- writing the e-mail ----
  const answersPayload = () => included.map((q) => ({ question: q.question, answer: q.answer.trim() }))

  const write = async () => {
    setBusy('writing')
    setError(null)
    try {
      const ctx = await ensureEntities()
      const created = await createOutreach({
        ...ctx,
        personal_note: note.trim() || null,
        answers: answersPayload(),
      })
      if (created.status === 'failed') throw new Error(created.error || 'The e-mail could not be written. Try again.')
      setEmail(created)
      setSubject(created.subject ?? '')
      setBody(created.body ?? '')
      setSubjectBase(created.subject ?? '')
      setBodyBase(created.body ?? '')
      setVariants({ subject: 0, body: 0 })
      setAttachment(null)
      update({ emailId: created.id })
      if (canAttach) void loadAttachment(created.id)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const loadAttachment = async (emailId: string): Promise<AttachmentInfo | null> => {
    try {
      const info = await emailAttachmentInfo(emailId, flow.template)
      setAttachment(info)
      return info
    } catch (err) {
      setAttachment(null)
      setError(err instanceof ChooseTemplateError ? `${err.message} Go back and choose another design.` : errorText(err))
      return null
    }
  }

  // ---- regenerating one part ----
  const subjectEdited = subject.trim() !== subjectBase.trim()
  const bodyEdited = body.trim() !== bodyBase.trim()
  const regenerate = async (part: 'subject' | 'body') => {
    if (!email) return
    const edited = part === 'subject' ? subjectEdited : bodyEdited
    if (edited && !window.confirm(`Replace your edits to the ${part} with a new one?`)) return
    setRegenerating(part)
    setError(null)
    try {
      const variant = variants[part] + 1
      const out = await regenerateEmailPart(email.id, {
        part, variant, ...jobContext,
        ...(part === 'body' ? { personal_note: note.trim() || null, answers: answersPayload() } : {}),
      })
      // only the asked-for part changes; the other part and everything else the user typed stays as it is
      if (part === 'subject' && out.subject) {
        setSubject(out.subject)
        setSubjectBase(out.subject)
      } else if (part === 'body' && out.body) {
        setBody(out.body)
        setBodyBase(out.body)
      } else {
        throw new Error(`There is no new ${part} to suggest right now.`)
      }
      setVariants((v) => ({ ...v, [part]: variant }))
    } catch (err) {
      setError(errorText(err))
    } finally {
      setRegenerating(null)
    }
  }

  const dirty = !!email && (subject !== (email.subject ?? '') || body !== (email.body ?? ''))
  const save = async () => {
    if (!email) return email
    if (!dirty) return email
    setBusy('saving')
    setError(null)
    try {
      const saved = await editEmail(email.id, { subject, body })
      setEmail(saved)
      setSubject(saved.subject ?? '')
      setBody(saved.body ?? '')
      return saved
    } catch (err) {
      setError(errorText(err))
      return null
    } finally {
      setBusy(null)
    }
  }

  const showAttachment = async () => {
    if (!email) return
    setBusy('preview')
    setError(null)
    try {
      const result = await emailAttachment(email.id, flow.template)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = result.url
      setPdf(result)
      await loadAttachment(email.id)
    } catch (err) {
      setError(err instanceof ChooseTemplateError ? `${err.message} Go back and choose another design.` : errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const send = async () => {
    if (!email) return
    const to = recipientEmail.trim()
    let withoutAttachment = false
    if (email.resume_draft_id) {
      // the exact file that will go out, checked now, shown in the confirmation
      setBusy('preview')
      const info = await loadAttachment(email.id)
      setBusy(null)
      if (!info?.has_attachment || !info.filename) return
      if (!window.confirm(`Send this e-mail to ${to} now, with “${info.filename}” attached? This cannot be undone.`)) return
    } else {
      if (!window.confirm('Your resume is not approved, so this e-mail would go out WITHOUT any attachment. Send it without a resume?')) return
      if (!window.confirm(`Final check: send this e-mail to ${to} with no attachment? This cannot be undone.`)) return
      withoutAttachment = true
    }
    const saved = await save()
    if (!saved) return
    await deliver({ withoutAttachment })
  }

  const deliver = async (opts: { withoutAttachment?: boolean; confirmRetry?: boolean }) => {
    if (!email) return
    setBusy('sending')
    setError(null)
    try {
      const done = await sendEmail(email.id, flow.template, opts)
      setEmail(done)
      if (done.status === 'failed') throw new Error(done.error || 'The e-mail could not be sent.')
    } catch (err) {
      if (actionOf(err) === 'confirm_retry' && window.confirm(`${errorText(err)}`)) {
        setBusy(null)
        return deliver({ ...opts, confirmRetry: true })
      }
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const discard = async () => {
    if (!email || !window.confirm('Discard this e-mail?')) return
    try {
      await rejectEmail(email.id)
    } catch {
      /* it may already be gone; either way start fresh */
    }
    setEmail(null)
    setPdf(null)
    setAttachment(null)
    update({ emailId: null })
  }

  if (sent) {
    return (
      <section className="space-y-6">
        <StepHeader title="Sent" />
        <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-2xl p-6 text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto shadow-lg shadow-emerald-500/20">
            <CheckCircle2 size={36} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white m-0">Outreach Delivered!</h3>
            <p className="text-sm text-slate-300 mt-2 max-w-md mx-auto">
              Your e-mail was sent to <strong className="text-white">{recipientEmail}</strong>
              {attachment?.has_attachment && attachment.filename ? (
                <> with <strong className="text-white">{attachment.filename}</strong> attached.</>
              ) : (
                <> without an attachment.</>
              )}
            </p>
          </div>
          <div className="pt-2">
            <button className="btn primary mx-auto" onClick={onRestart}>
              <RotateCcw size={15} />
              Apply to another job
            </button>
          </div>
        </div>
      </section>
    )
  }

  if (!email) {
    return (
      <section className="space-y-6">
        <StepHeader
          title="Write to the founder"
          intro="We write a short e-mail from your real experience. You review and edit it before anything is sent."
        />

        {!approved && (
          <Notice tone="warn">
            Your tailored resume is not approved, so this e-mail would go out without a resume attached. You can still write it; sending
            without an attachment needs one more confirmation. The original upload is never attached.
          </Notice>
        )}
        {approved && !flow.template && (
          <Notice tone="warn">Choose a design first so your resume can be attached when you send.</Notice>
        )}

        {/* Company & Recipient */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0 flex items-center gap-2">
            <Mail size={15} /> Recipient Information
          </h3>
          <div className="grid2">
            <TextInput label="Company" value={company} onChange={setCompany} />
            <TextInput
              label="Company website (optional)"
              value={website}
              onChange={setWebsite}
              hint="Read once, over HTTPS, to find something real to mention."
            />
            <TextInput
              label="Recipient's name (optional)"
              value={recipientName}
              onChange={setRecipientName}
              hint="Without a name the greeting is “Hi {company} Team,”."
            />
            <TextInput label="Recipient's e-mail" type="email" value={recipientEmail} onChange={setRecipientEmail} />
            <TextInput label="Their role (optional)" value={role} onChange={setRole} />
          </div>
        </div>

        {/* Personal line */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0 flex items-center gap-2">
            <Sparkles size={15} /> Personalization
          </h3>
          <TextArea
            label="One personal line (optional)"
            rows={2}
            maxLength={240}
            value={note}
            onChange={setNote}
            hint="Why this company? Used exactly as written, at the start of the intro."
            actions={
              <>
                {note.trim() && <Source edited={noteEdited} />}
                {noteEdited && noteBase && (
                  <button type="button" className="btn small ghost" disabled={noteBusy} onClick={() => setNote(noteBase)}>
                    <Undo2 size={13} aria-hidden /> Restore suggestion
                  </button>
                )}
                <button
                  type="button"
                  className="btn small"
                  disabled={noteBusy || !ready}
                  onClick={() => void writeNote()}
                  aria-label={noteInfo ? 'Regenerate personal line' : 'Suggest a personal line'}
                >
                  <RefreshCw size={13} aria-hidden className={noteBusy ? 'animate-spin' : ''} />{' '}
                  {noteBusy ? 'Working…' : noteInfo ? 'Regenerate' : 'Suggest one'}
                </button>
              </>
            }
          />
          <ErrorBanner message={noteError} />
          {noteInfo?.limitation && <Notice tone="warn">{noteInfo.limitation}</Notice>}
          {noteInfo && !noteInfo.limitation && noteInfo.basis !== 'none' && noteInfo.evidence[0] && (
            <p className="muted small">
              Based on {noteInfo.basis === 'website' ? 'the company’s website' : 'the job description'}: “{noteInfo.evidence[0]}”
            </p>
          )}
          {!ready && <p className="muted small">Fill in the company and the recipient’s e-mail to get suggestions.</p>}
        </div>

        {/* Questions */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0 flex items-center gap-2">
            <Wand2 size={15} /> Questions they asked (optional)
          </h3>
          <p className="muted small m-0">
            Each answer is at most {ANSWER_WORDS} words, with proof from your resume. Answers stay separate from the e-mail unless you add them.
          </p>
          <div className="chips pt-1">
            {[...SUGGESTED, ...fromJob].filter((q) => !qas.some((x) => x.question === q)).map((q) => (
              <button key={q} type="button" className="chip" onClick={() => addQuestion(q)} disabled={qas.length >= MAX_QUESTIONS}>
                + {q}
              </button>
            ))}
          </div>
          {fromJob.length > 0 && <p className="muted small m-0">Questions found in the job description are listed above.</p>}
          {qas.map((qa) => (
            <QuestionCard
              key={qa.question}
              qa={qa}
              onChange={(patch) => patchQa(qa.question, patch)}
              onRegenerate={() => regenerateAnswer(qa)}
              onGenerate={() => void generateAnswer(qa.question, qa.variant)}
              onRemove={() => removeQuestion(qa.question)}
              ready={ready}
            />
          ))}
          <CustomQuestion onAdd={addQuestion} disabled={qas.length >= MAX_QUESTIONS} />
        </div>

        {problems.map((p) => (
          <p key={p} className="error-text">
            {p}
          </p>
        ))}
        {badAnswer && <p className="error-text">An answer in the e-mail can be at most {ANSWER_WORDS} words.</p>}

        {busy === 'writing' && <Working label="Writing your e-mail" hint="Usually under half a minute." />}
        <ErrorBanner message={error} onRetry={write} />

        <div className="actions">
          <button className="btn primary" disabled={busy !== null || !ready || !!badAnswer} onClick={write}>
            <Sparkles size={16} />
            Write my e-mail
          </button>
        </div>
      </section>
    )
  }

  const words = bodyWords(body)
  const failed = email.status === 'failed'
  return (
    <section className="space-y-6">
      <StepHeader
        title="Review your e-mail"
        intro={`To ${recipientEmail.trim()}. Edit freely: what you see here is exactly what is sent. Nothing is sent until you confirm.`}
      />

      {failed && (
        <Notice tone="warn">
          The last attempt to send failed{email.error ? `: ${email.error}` : ''}. Your e-mail is kept. Check that it did not reach {recipientEmail.trim()},
          then send again.
        </Notice>
      )}

      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <TextInput
          label="Subject"
          value={subject}
          onChange={setSubject}
          maxLength={150}
          actions={
            <>
              <Source edited={subjectEdited} />
              {subjectEdited && (
                <button type="button" className="btn small ghost" disabled={busy !== null} onClick={() => setSubject(subjectBase)}>
                  <Undo2 size={13} aria-hidden /> Restore suggestion
                </button>
              )}
              <button
                type="button"
                className="btn small"
                disabled={busy !== null || regenerating !== null}
                onClick={() => void regenerate('subject')}
                aria-label="Regenerate subject"
              >
                <RefreshCw size={13} aria-hidden className={regenerating === 'subject' ? 'animate-spin' : ''} />{' '}
                {regenerating === 'subject' ? 'Working…' : 'Regenerate'}
              </button>
            </>
          }
        />
        <TextArea
          label="E-mail"
          value={body}
          onChange={setBody}
          rows={16}
          maxLength={8000}
          actions={
            <>
              <Source edited={bodyEdited} />
              {bodyEdited && (
                <button type="button" className="btn small ghost" disabled={busy !== null} onClick={() => setBody(bodyBase)}>
                  <Undo2 size={13} aria-hidden /> Restore suggestion
                </button>
              )}
              <button
                type="button"
                className="btn small"
                disabled={busy !== null || regenerating !== null}
                onClick={() => void regenerate('body')}
                aria-label="Regenerate body"
              >
                <RefreshCw size={13} aria-hidden className={regenerating === 'body' ? 'animate-spin' : ''} />{' '}
                {regenerating === 'body' ? 'Working…' : 'Regenerate'}
              </button>
            </>
          }
          hint={`${words} words before the signature (aim for ${WORDS_MIN}–${WORDS_MAX}).`}
        />
        {(words < WORDS_MIN || words > WORDS_MAX) && words > 0 && (
          <p className="muted small m-0">
            {words < WORDS_MIN
              ? 'A little short: add a sentence of your own, or regenerate for a fuller draft.'
              : 'A little long: shorter e-mails get read more often.'}
          </p>
        )}

        {/* Attachment */}
        {email.resume_draft_id ? (
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-300">
            <Paperclip size={14} className="text-indigo-400" />
            <span>
              Will attach:{' '}
              <strong className="text-white">{attachment?.filename ?? 'checking the file…'}</strong>
              {flow.template ? ` (${flow.template === 'original' ? 'your own design' : `${flow.template} template`})` : ''}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-300">
            <AlertTriangle size={14} />
            <span>No resume will be attached: your tailored resume is not approved. You will be asked to confirm.</span>
          </div>
        )}
      </div>

      <ErrorBanner message={error} />

      {busy && <Working label={busy === 'sending' ? 'Sending' : busy === 'preview' ? 'Building the attachment' : 'Saving'} />}

      {pdf && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <FileCheck size={16} className="text-emerald-400" />
            PDF Attachment Preview
          </div>
          <iframe className="pdf rounded-xl" title="E-mail attachment preview" src={pdf.url} />
        </div>
      )}

      <div className="actions">
        {email.resume_draft_id && (
          <button className="btn" disabled={busy !== null} onClick={showAttachment}>
            <Paperclip size={15} />
            {pdf ? 'Refresh attachment' : 'Preview attachment'}
          </button>
        )}
        <button className="btn" disabled={busy !== null || !dirty} onClick={() => void save()}>
          Save edits
        </button>
        <button
          className="btn primary"
          disabled={busy !== null || regenerating !== null || !subject.trim() || !body.trim() || (!!email.resume_draft_id && !flow.template)}
          onClick={send}
        >
          <Send size={15} />
          {busy === 'sending' ? 'Sending…' : failed ? 'Try sending again' : 'Send now'}
        </button>
        <button className="btn ghost" disabled={busy !== null} onClick={discard}>
          <Trash2 size={14} className="mr-1 inline text-rose-400" />
          Discard
        </button>
        {dirty && <Badge tone="warn">unsaved edits</Badge>}
      </div>
    </section>
  )
}

function QuestionCard({ qa, onChange, onRegenerate, onGenerate, onRemove, ready }: {
  qa: QA
  onChange: (patch: Partial<QA>) => void
  onRegenerate: () => void
  onGenerate: () => void
  onRemove: () => void
  ready: boolean
}) {
  const edited = qa.answer.trim() !== qa.base.trim()
  const words = wordsOf(qa.answer)
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 space-y-3" aria-label={`Question: ${qa.question}`}>
      <div className="flex items-start justify-between gap-3">
        <p className="m-0 text-sm font-medium text-white">{qa.question}</p>
        <button type="button" className="btn small ghost shrink-0" onClick={onRemove} aria-label={`Remove question ${qa.question}`}>
          <Trash2 size={13} aria-hidden />
        </button>
      </div>
      <TextArea
        label="Answer"
        rows={2}
        value={qa.answer}
        onChange={(v) => onChange({ answer: v })}
        hint={`${words}/${ANSWER_WORDS} words`}
        actions={
          <>
            {qa.answer.trim() && <Source edited={edited} />}
            {edited && qa.base && (
              <button type="button" className="btn small ghost" disabled={qa.busy} onClick={() => onChange({ answer: qa.base })}>
                <Undo2 size={13} aria-hidden /> Restore suggestion
              </button>
            )}
            <button
              type="button"
              className="btn small"
              disabled={qa.busy || !ready}
              onClick={qa.base ? onRegenerate : onGenerate}
              aria-label={`${qa.base ? 'Regenerate' : 'Write'} answer to ${qa.question}`}
            >
              <RefreshCw size={13} aria-hidden className={qa.busy ? 'animate-spin' : ''} /> {qa.busy ? 'Working…' : qa.base ? 'Regenerate' : 'Write answer'}
            </button>
          </>
        }
      />
      {words > ANSWER_WORDS && <p className="error-text">An answer can be at most {ANSWER_WORDS} words.</p>}
      {qa.error && <p className="error-text" role="alert">{qa.error}</p>}
      {qa.evidence.length > 0 && (
        <div className="space-y-1">
          <span className="muted small">From your resume:</span>
          <ul className="m-0 pl-5 text-xs text-slate-300 space-y-0.5">
            {qa.evidence.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}
      <label className="flex items-center gap-2 text-xs text-slate-300">
        <input type="checkbox" checked={qa.include} disabled={!qa.answer.trim()} onChange={(e) => onChange({ include: e.target.checked })} />
        Add this answer to the e-mail (otherwise it stays here, ready to paste into a reply)
      </label>
    </div>
  )
}

function CustomQuestion({ onAdd, disabled }: { onAdd: (q: string) => void; disabled: boolean }) {
  const [text, setText] = useState('')
  const add = () => {
    const q = text.trim().replace(/\s+/g, ' ')
    if (q.length >= 3 && q.length <= 200) {
      onAdd(q)
      setText('')
    }
  }
  return (
    <div className="inline flex gap-2">
      <input
        aria-label="Another question"
        placeholder="Another question they asked"
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            add()
          }
        }}
      />
      <button type="button" className="btn small shrink-0" onClick={add} disabled={disabled || text.trim().length < 3}>
        <Plus size={13} className="mr-1 inline" />
        Add
      </button>
    </div>
  )
}
