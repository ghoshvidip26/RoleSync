import { useRef, useState } from 'react'
import {
  ArrowRight,
  CheckCircle2,
  FileCheck2,
  FileText,
  RotateCcw,
  Sparkles,
  UploadCloud,
  UserCheck,
} from 'lucide-react'
import { createProfile, extractProfile, uploadOriginalResume } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { cleanProfile, errorText, PLACEHOLDER_EMAIL, profileProblems } from '../lib.ts'
import type { OriginalAssessment, ProfileData } from '../types.ts'
import {
  Badge,
  EntryList,
  ErrorBanner,
  Notice,
  StepHeader,
  TagsInput,
  TextArea,
  TextInput,
  Working,
} from '../ui.tsx'

const MAX_BYTES = 5 * 1024 * 1024

export function UploadStep({ onNext }: { onNext: () => void }) {
  const { flow, update, reset } = useFlow()
  const [file, setFile] = useState<File | null>(null)
  const [form, setForm] = useState<ProfileData | null>(null)
  const [busy, setBusy] = useState<'reading' | 'saving' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
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
        skills: profile.skills ?? [],
        programming_languages: profile.programming_languages ?? [],
        frameworks: profile.frameworks ?? [],
        databases_tools: profile.databases_tools ?? [],
        experience: profile.experience ?? [],
        projects: profile.projects ?? [],
        achievements: profile.achievements ?? [],
        education: profile.education ?? [],
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
          setNote(
            `Your profile is saved, but your resume's own design could not be stored (${errorText(
              err
            )}). You can still use the template gallery.`
          )
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
      <section className="space-y-6">
        <StepHeader
          title="Your profile"
          intro="Your resume has been parsed and indexed. You can proceed to target a specific job."
        />

        {/* Profile Summary Card */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3.5">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-emerald-500/20">
                {flow.profile.name.charAt(0).toUpperCase()}
              </div>
              <div>
                <h3 className="text-base font-semibold text-white m-0">
                  {flow.profile.name}
                </h3>
                <p className="text-xs text-slate-400 m-0">{flow.profile.email}</p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {flow.profile.github && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      GitHub: {flow.profile.github}
                    </span>
                  )}
                  {flow.profile.linkedin && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      LinkedIn: {flow.profile.linkedin}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <UserCheck size={13} /> Saved & Indexed
            </span>
          </div>
        </div>

        <Notice tone="good">
          Saved as <strong>{flow.profile.name}</strong> ({flow.profile.email}).
        </Notice>
        {note && <Notice tone="warn">{note}</Notice>}
        {a?.preserve_supported && (
          <Notice tone="good">
            Your resume's own design can be reused for the tailored PDF.
          </Notice>
        )}
        {a && !a.preserve_supported && (
          <Notice tone="warn">
            Your resume's own design can't be reproduced reliably
            {a.reason ? `: ${a.reason}` : ''}. You will pick one of our ATS-safe templates instead.
          </Notice>
        )}
        {a?.ats && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
            <span className="text-sm text-slate-300">ATS risk of your original:</span>
            <Badge tone={a.ats.risk === 'low' ? 'good' : a.ats.risk === 'medium' ? 'warn' : 'bad'}>
              {a.ats.risk}
            </Badge>
          </div>
        )}
        {a?.ats?.flags.map((f) => (
          <p key={f.code} className="small muted">
            • {f.message}
          </p>
        ))}

        <div className="actions">
          <button className="btn primary" onClick={onNext}>
            Continue <ArrowRight size={15} />
          </button>
          <button
            className="btn ghost"
            onClick={() => {
              reset()
              setFile(null)
            }}
          >
            <RotateCcw size={14} className="mr-1 inline" />
            Start over with another resume
          </button>
        </div>
      </section>
    )
  }

  const set = <K extends keyof ProfileData>(key: K, value: ProfileData[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f))
  const problems = form ? profileProblems(form) : []

  return (
    <section className="space-y-6">
      <StepHeader
        title="Upload your resume"
        intro="We read your PDF, then you check what we found. Your original file is never changed."
      />

      {/* Modern Interactive Dropzone */}
      <div
        className={`drop relative transition-all duration-200 ${
          isDragging ? 'active border-indigo-500 bg-indigo-500/10' : ''
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          choose(e.dataTransfer.files[0])
        }}
        onClick={() => input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          aria-label="Resume PDF"
          onChange={(e) => choose(e.target.files?.[0])}
        />

        <div className="flex flex-col items-center text-center p-4">
          <div className="w-14 h-14 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-3 shadow-inner">
            {file ? <FileCheck2 size={28} className="text-emerald-400" /> : <UploadCloud size={28} />}
          </div>

          <div className="space-y-1 mb-4">
            <h4 className="text-base font-semibold text-white">
              {file ? file.name : 'Choose a PDF or drag & drop here'}
            </h4>
            <p className="text-xs text-slate-400">
              {file
                ? `${Math.round(file.size / 1024)} KB · Click to replace file`
                : 'Supports PDF format up to 5 MB'}
            </p>
          </div>

          <button
            type="button"
            className="btn primary shadow-md"
            onClick={(e) => {
              e.stopPropagation()
              input.current?.click()
            }}
          >
            <FileText size={15} />
            Choose a PDF
          </button>

          <span className="muted text-xs mt-3 block">
            {file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : 'or drop it here · PDF, up to 5 MB'}
          </span>
        </div>
      </div>

      <ErrorBanner message={error} onRetry={file && !form ? read : undefined} />

      {busy === 'reading' && (
        <Working label="Reading your resume" hint="Extracting text, sections, and candidate profile data..." />
      )}

      {file && !form && busy !== 'reading' && (
        <div className="actions">
          <button className="btn primary" onClick={read}>
            <Sparkles size={16} />
            Read my resume
          </button>
        </div>
      )}

      {form && (
        <form
          className="stack space-y-6"
          onSubmit={(e) => {
            e.preventDefault()
            void save()
          }}
        >
          <Notice>
            Check everything below. Fix anything we got wrong. This is what tailoring and emails are built from.
          </Notice>
          {form.email.trim().toLowerCase() === PLACEHOLDER_EMAIL && (
            <Notice tone="warn">We could not find an e-mail address in your resume. Enter yours.</Notice>
          )}

          {/* Contact Details Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0">
              Personal & Contact Information
            </h3>
            <div className="grid2">
              <TextInput label="Name" value={form.name} onChange={(v) => set('name', v)} />
              <TextInput label="E-mail" type="email" value={form.email} onChange={(v) => set('email', v)} />
              <TextInput label="GitHub" value={form.github ?? ''} onChange={(v) => set('github', v)} />
              <TextInput label="LinkedIn" value={form.linkedin ?? ''} onChange={(v) => set('linkedin', v)} />
              <TextInput label="Portfolio" value={form.portfolio ?? ''} onChange={(v) => set('portfolio', v)} />
            </div>
            <TextArea label="Summary" value={form.summary ?? ''} onChange={(v) => set('summary', v)} rows={3} />
          </div>

          {/* Skills Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0">
              Skills & Tech Stack
            </h3>
            <TagsInput
              label="Languages"
              value={form.programming_languages}
              onChange={(v) => set('programming_languages', v)}
            />
            <TagsInput label="Frameworks" value={form.frameworks} onChange={(v) => set('frameworks', v)} />
            <TagsInput
              label="Databases and tools"
              value={form.databases_tools}
              onChange={(v) => set('databases_tools', v)}
            />
            <TagsInput label="All skills" value={form.skills} onChange={(v) => set('skills', v)} />
          </div>

          {/* Experience & Projects */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-400 m-0">
              Experience & Projects
            </h3>
            <EntryList
              label="Experience"
              items={form.experience}
              onChange={(v) => set('experience', v)}
              rows={7}
              addLabel="Add a role"
              hint="One box per role: “Title | Company, Location”, then the dates, then one bullet per line starting with •"
            />
            <EntryList
              label="Projects"
              items={form.projects}
              onChange={(v) => set('projects', v)}
              rows={6}
              addLabel="Add a project"
            />
            <EntryList
              label="Achievements"
              items={form.achievements}
              onChange={(v) => set('achievements', v)}
              rows={2}
              addLabel="Add an achievement"
            />
            <EntryList
              label="Education"
              items={form.education}
              onChange={(v) => set('education', v)}
              rows={2}
              addLabel="Add education"
            />
          </div>

          {problems.map((p) => (
            <p key={p} className="error-text">
              {p}
            </p>
          ))}

          {busy === 'saving' && (
            <Working
              label="Saving your profile"
              hint="Building the search index for your resume can take a little while."
            />
          )}

          <div className="actions">
            <button
              className="btn primary"
              type="submit"
              disabled={busy !== null || problems.length > 0}
            >
              <CheckCircle2 size={16} />
              Confirm and save
            </button>
          </div>
        </form>
      )}
    </section>
  )
}
