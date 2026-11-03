import { useRef, useState } from 'react'
import { createProfile, extractProfile, uploadOriginalResume } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { cleanProfile, errorText, PLACEHOLDER_EMAIL, profileProblems } from '../lib.ts'
import type { OriginalAssessment, ProfileData } from '../types.ts'
import { Badge, EntryList, ErrorBanner, Notice, StepHeader, TagsInput, TextArea, TextInput, Working } from '../ui.tsx'

const MAX_BYTES = 5 * 1024 * 1024

export function UploadStep({ onNext }: { onNext: () => void }) {
  const { flow, update, reset } = useFlow()
  const [file, setFile] = useState<File | null>(null)
  const [form, setForm] = useState<ProfileData | null>(null)
  const [busy, setBusy] = useState<'reading' | 'saving' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const choose = (f: File | undefined) => {
    setError(null)
    if (!f) return
    if (!/\.pdf$/i.test(f.name) && f.type !== 'application/pdf') return setError('Choose a PDF file.')
    if (f.size > MAX_BYTES) return setError('The resume must be 5 MB or smaller.')
    if (f.size === 0) return setError('That file is empty.')
    setFile(f)
    setForm(null)
  }

  const read = async () => {
    if (!file) return
    setBusy('reading')
    setError(null)
    try {
      const extracted = await extractProfile(file)
      const { raw_resume_text: _raw, ...profile } = extracted
      setForm({
        ...profile,
        skills: profile.skills ?? [], programming_languages: profile.programming_languages ?? [],
        frameworks: profile.frameworks ?? [], databases_tools: profile.databases_tools ?? [],
        experience: profile.experience ?? [], projects: profile.projects ?? [],
        achievements: profile.achievements ?? [], education: profile.education ?? [],
      })
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const save = async () => {
    if (!form) return
    setBusy('saving')
    setError(null)
    setNote(null)
    try {
      const profile = await createProfile(cleanProfile(form))
      let assessment: OriginalAssessment | null = null
      if (file) {
        try {
          assessment = await uploadOriginalResume(profile.id, file)
        } catch (err) {
          setNote(`Your profile is saved, but your resume's own design could not be stored (${errorText(err)}). You can still use the template gallery.`)
        }
      }
      update({ profile, assessment, draft: null, fit: null, template: null, emailId: null })
      setForm(null)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  if (flow.profile && !form) {
    const a = flow.assessment
    return (
      <section>
        <StepHeader title="Your profile" />
        <Notice tone="good">Saved as <strong>{flow.profile.name}</strong> ({flow.profile.email}).</Notice>
        {note && <Notice tone="warn">{note}</Notice>}
        {a?.preserve_supported && <Notice tone="good">Your resume's own design can be reused for the tailored PDF.</Notice>}
        {a && !a.preserve_supported && (
          <Notice tone="warn">
            Your resume's own design can't be reproduced reliably{a.reason ? `: ${a.reason}` : ''}. You will pick one of our ATS-safe templates instead.
          </Notice>
        )}
        {a?.ats && (
          <p>ATS risk of your original: <Badge tone={a.ats.risk === 'low' ? 'good' : a.ats.risk === 'medium' ? 'warn' : 'bad'}>{a.ats.risk}</Badge></p>
        )}
        {a?.ats?.flags.map((f) => <p key={f.code} className="small muted">• {f.message}</p>)}
        <div className="actions">
          <button className="btn primary" onClick={onNext}>Continue</button>
          <button className="btn ghost" onClick={() => { reset(); setFile(null) }}>Start over with another resume</button>
        </div>
      </section>
    )
  }

  const set = <K extends keyof ProfileData>(key: K, value: ProfileData[K]) => setForm((f) => (f ? { ...f, [key]: value } : f))
  const problems = form ? profileProblems(form) : []

  return (
    <section>
      <StepHeader title="Upload your resume" intro="We read your PDF, then you check what we found. Your original file is never changed." />
      <div className="drop"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files[0]) }}>
        <input ref={input} type="file" accept="application/pdf,.pdf" hidden aria-label="Resume PDF" onChange={(e) => choose(e.target.files?.[0])} />
        <button type="button" className="btn" onClick={() => input.current?.click()}>Choose a PDF</button>
        <span className="muted">{file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : 'or drop it here · PDF, up to 5 MB'}</span>
      </div>
      <ErrorBanner message={error} onRetry={file && !form ? read : undefined} />
      {busy === 'reading' && <Working label="Reading your resume" hint="Usually a few seconds." />}
      {file && !form && busy !== 'reading' && <div className="actions"><button className="btn primary" onClick={read}>Read my resume</button></div>}

      {form && (
        <form className="stack" onSubmit={(e) => { e.preventDefault(); void save() }}>
          <Notice>Check everything below. Fix anything we got wrong. This is what tailoring and emails are built from.</Notice>
          {form.email.trim().toLowerCase() === PLACEHOLDER_EMAIL && <Notice tone="warn">We could not find an e-mail address in your resume. Enter yours.</Notice>}
          <div className="grid2">
            <TextInput label="Name" value={form.name} onChange={(v) => set('name', v)} />
            <TextInput label="E-mail" type="email" value={form.email} onChange={(v) => set('email', v)} />
            <TextInput label="GitHub" value={form.github ?? ''} onChange={(v) => set('github', v)} />
            <TextInput label="LinkedIn" value={form.linkedin ?? ''} onChange={(v) => set('linkedin', v)} />
            <TextInput label="Portfolio" value={form.portfolio ?? ''} onChange={(v) => set('portfolio', v)} />
          </div>
          <TextArea label="Summary" value={form.summary ?? ''} onChange={(v) => set('summary', v)} rows={3} />
          <TagsInput label="Languages" value={form.programming_languages} onChange={(v) => set('programming_languages', v)} />
          <TagsInput label="Frameworks" value={form.frameworks} onChange={(v) => set('frameworks', v)} />
          <TagsInput label="Databases and tools" value={form.databases_tools} onChange={(v) => set('databases_tools', v)} />
          <TagsInput label="All skills" value={form.skills} onChange={(v) => set('skills', v)} />
          <EntryList label="Experience" items={form.experience} onChange={(v) => set('experience', v)} rows={7} addLabel="Add a role"
            hint="One box per role: “Title | Company, Location”, then the dates, then one bullet per line starting with •" />
          <EntryList label="Projects" items={form.projects} onChange={(v) => set('projects', v)} rows={6} addLabel="Add a project" />
          <EntryList label="Achievements" items={form.achievements} onChange={(v) => set('achievements', v)} rows={2} addLabel="Add an achievement" />
          <EntryList label="Education" items={form.education} onChange={(v) => set('education', v)} rows={2} addLabel="Add education" />
          {problems.map((p) => <p key={p} className="error-text">{p}</p>)}
          {busy === 'saving' && <Working label="Saving your profile" hint="Building the search index for your resume can take a little while." />}
          <div className="actions">
            <button className="btn primary" type="submit" disabled={busy !== null || problems.length > 0}>Confirm and save</button>
          </div>
        </form>
      )}
    </section>
  )
}
