import { useMemo, useState } from 'react'
import {
  ArrowRight,
  CheckCircle2,
  RotateCcw,
  Save,
  Wand2,
} from 'lucide-react'
import { approveDraft, editDraft, rejectDraft, tailorResume } from '../api.ts'
import { PositioningSection } from '../components/PositioningSection.tsx'
import { useFlow } from '../flow.tsx'
import { draftChanges, errorText, parseLines, problemsIn, profileSkillList } from '../lib.ts'
import type { DraftRecord, ExperienceItem, ProjectItem, ResumeDraft } from '../types.ts'
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

/** What the API wants: blank optional text becomes null. */
function normalize(d: ResumeDraft): ResumeDraft {
  const nul = (s: string | null) => (s && s.trim() ? s.trim() : null)
  return {
    ...d,
    headline: nul(d.headline),
    summary: nul(d.summary),
    experience: d.experience.map((e) => ({
      ...e,
      company: e.company.trim(),
      job_title: e.job_title.trim(),
      duration: nul(e.duration),
      location: nul(e.location),
      bullets: e.bullets.map((b) => b.trim()).filter(Boolean),
    })),
    projects: d.projects.map((p) => ({
      ...p,
      name: p.name.trim(),
      description: nul(p.description),
      bullets: p.bullets.map((b) => b.trim()).filter(Boolean),
    })),
    achievements: d.achievements.map((a) => a.trim()).filter(Boolean),
    skills: d.skills.map((s) => s.trim()).filter(Boolean),
  }
}

/** Enough rows to show every bullet in full (about 95 characters fit on a line). */
const bulletRows = (bullets: string[]) =>
  Math.max(3, bullets.reduce((n, b) => n + Math.ceil((b.length || 1) / 95), 0) + 1)

function ExperienceEditor({
  items,
  onChange,
  locked,
}: {
  items: ExperienceItem[]
  onChange: (v: ExperienceItem[]) => void
  locked: boolean
}) {
  const patch = (i: number, c: Partial<ExperienceItem>) =>
    onChange(items.map((e, j) => (j === i ? { ...e, ...c } : e)))
  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold text-white flex items-center gap-2 m-0">
        Work Experience
      </h3>
      {items.map((e, i) => (
        <fieldset key={i} className="entries space-y-3" disabled={locked}>
          <legend className="text-indigo-400 font-semibold">
            {e.job_title || 'Role'} · {e.company || 'Company'}
          </legend>
          <div className="grid2">
            <TextInput label="Job title" value={e.job_title} onChange={(v) => patch(i, { job_title: v })} />
            <TextInput label="Company" value={e.company} onChange={(v) => patch(i, { company: v })} />
            <TextInput label="Dates" value={e.duration ?? ''} onChange={(v) => patch(i, { duration: v })} />
            <TextInput label="Location" value={e.location ?? ''} onChange={(v) => patch(i, { location: v })} />
          </div>
          <TextArea
            label="Bullets"
            hint="One bullet per line"
            rows={bulletRows(e.bullets)}
            value={e.bullets.join('\n')}
            onChange={(v) => patch(i, { bullets: parseLines(v) })}
          />
        </fieldset>
      ))}
    </div>
  )
}

function ProjectEditor({
  items,
  onChange,
  locked,
}: {
  items: ProjectItem[]
  onChange: (v: ProjectItem[]) => void
  locked: boolean
}) {
  const patch = (i: number, c: Partial<ProjectItem>) =>
    onChange(items.map((p, j) => (j === i ? { ...p, ...c } : p)))
  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold text-white flex items-center gap-2 m-0">
        Key Projects
      </h3>
      {items.map((p, i) => (
        <fieldset key={i} className="entries space-y-3" disabled={locked}>
          <legend className="text-indigo-400 font-semibold">{p.name || 'Project'}</legend>
          <TextInput label="Name" value={p.name} onChange={(v) => patch(i, { name: v })} />
          <TagsInput
            label="Technologies"
            value={p.technologies}
            onChange={(v) => patch(i, { technologies: v })}
          />
          <TextArea
            label="Bullets"
            hint="One bullet per line"
            rows={bulletRows(p.bullets)}
            value={p.bullets.join('\n')}
            onChange={(v) => patch(i, { bullets: parseLines(v) })}
          />
        </fieldset>
      ))}
    </div>
  )
}

export function DraftStep({ onNext }: { onNext: () => void }) {
  const { flow, update } = useFlow()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generate = async () => {
    if (!flow.profile) return
    setBusy(true)
    setError(null)
    try {
      const out = await tailorResume(flow.profile.id, flow.job.description.trim(), flow.job.title)
      if (!out.draft_id) throw new Error('The draft could not be saved. Try again.')
      update({
        draft: {
          id: out.draft_id,
          profile_id: flow.profile.id,
          status: 'pending_review',
          draft: out.draft,
        },
        // what the AI wrote for the positioning fields, so the page can tell it apart from the user's edits
        positioning: {
          draftId: out.draft_id,
          headline: out.draft.headline ?? '',
          summary: out.draft.summary ?? '',
          skills: out.draft.skills ?? [],
          variants: { headline: 0, summary: 0, skills: 0 },
        },
        template: null,
        emailId: null,
      })
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  if (flow.draft) {
    return (
      <DraftEditor
        key={flow.draft.id}
        record={flow.draft}
        onNext={onNext}
        onRegenerate={generate}
        generating={busy}
      />
    )
  }
  return (
    <section className="space-y-6">
      <StepHeader
        title="Tailor your resume"
        intro="We rewrite your resume for this job using only what is already in it. You review and approve every change."
      />
      {busy && <Working label="Writing your tailored resume" hint="Usually under a minute." />}
      <ErrorBanner message={error} onRetry={generate} />
      {!busy && !error && (
        <div className="actions">
          <button className="btn primary" onClick={generate}>
            <Wand2 size={16} />
            Tailor my resume
          </button>
        </div>
      )}
    </section>
  )
}

function DraftEditor({
  record,
  onNext,
  onRegenerate,
  generating,
}: {
  record: DraftRecord
  onNext: () => void
  onRegenerate: () => Promise<void>
  generating: boolean
}) {
  const { flow, update } = useFlow()
  const [edited, setEdited] = useState<ResumeDraft>(record.draft)
  const [busy, setBusy] = useState<null | 'saving' | 'approving' | 'rejecting'>(null)
  const [error, setError] = useState<string | null>(null)

  const approved = record.status === 'approved'
  const clean = useMemo(() => normalize(edited), [edited])
  const changes = draftChanges(record.draft, clean)
  const dirty = Object.keys(changes).length > 0
  const problems = problemsIn(edited)

  const persist = async () => {
    if (!dirty) return
    update({ draft: await editDraft(record.id, changes) })
  }

  const save = async () => {
    setBusy('saving')
    setError(null)
    try {
      await persist()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const approve = async () => {
    setBusy('approving')
    setError(null)
    try {
      await persist()
      update({ draft: await approveDraft(record.id), template: null, emailId: null })
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const reject = async () => {
    if (!window.confirm('Discard this draft and write a new one?')) return
    setBusy('rejecting')
    setError(null)
    try {
      if (!approved) await rejectDraft(record.id)
      update({ draft: null, template: null, emailId: null })
      await onRegenerate()
    } catch (err) {
      setError(errorText(err))
      setBusy(null)
    }
  }

  const locked = approved || busy !== null
  const set = (c: Partial<ResumeDraft>) => setEdited({ ...edited, ...c })

  return (
    <section className="space-y-6">
      <StepHeader
        title="Review your tailored resume"
        intro="Edit anything. Nothing is used until you approve it."
      />

      {/* Draft Status Banner */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <span>Status:</span>
          {approved ? (
            <Badge tone="good">approved</Badge>
          ) : (
            <Badge tone="warn">waiting for your review</Badge>
          )}
        </div>
        {dirty && (
          <span className="text-xs text-amber-400 font-medium bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
            Unsaved changes
          </span>
        )}
      </div>

      {edited.claims_requiring_confirmation.length > 0 && (
        <Notice tone="warn">
          <strong>Add only if you have really used them:</strong>
          <ul className="mt-1 space-y-1">
            {edited.claims_requiring_confirmation.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </Notice>
      )}

      {edited.recommendations.length > 0 && (
        <Notice>
          <strong>Suggestions</strong>
          <ul className="mt-1 space-y-1">
            {edited.recommendations.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </Notice>
      )}

      {/* Editor Fields */}
      <div className="stack space-y-5">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <PositioningSection
            draftId={record.id}
            values={{ headline: edited.headline ?? '', summary: edited.summary ?? '', skills: edited.skills }}
            onChange={(patch) => set(patch)}
            profileSkills={profileSkillList(flow.profile)}
            jobTitle={flow.job.title}
            locked={locked}
          />
        </div>

        <ExperienceEditor items={edited.experience} onChange={(v) => set({ experience: v })} locked={locked} />
        <ProjectEditor items={edited.projects} onChange={(v) => set({ projects: v })} locked={locked} />

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
          <EntryList
            label="Achievements"
            items={edited.achievements}
            onChange={(v) => set({ achievements: v })}
            rows={2}
          />
        </div>

        {edited.education.length > 0 && (
          <p className="small muted">Education is kept exactly as in your resume.</p>
        )}
      </div>

      {problems.map((p) => (
        <p key={p} className="error-text">
          {p}
        </p>
      ))}

      <ErrorBanner message={error} />

      {busy && (
        <Working
          label={
            busy === 'approving' ? 'Approving' : busy === 'saving' ? 'Saving' : 'Discarding'
          }
        />
      )}
      {generating && <Working label="Writing a new draft" />}

      <div className="actions">
        {approved ? (
          <button className="btn primary" onClick={onNext}>
            Choose a design <ArrowRight size={15} />
          </button>
        ) : (
          <>
            <button
              className="btn primary"
              disabled={busy !== null || problems.length > 0}
              onClick={approve}
            >
              <CheckCircle2 size={16} />
              {dirty ? 'Save and approve' : 'Approve'}
            </button>
            <button
              className="btn"
              disabled={busy !== null || !dirty || problems.length > 0}
              onClick={save}
            >
              <Save size={15} />
              Save changes
            </button>
          </>
        )}
        <button
          className="btn ghost"
          disabled={busy !== null || generating}
          onClick={reject}
        >
          <RotateCcw size={14} className="mr-1 inline" />
          {approved ? 'Start a new draft' : 'Reject and rewrite'}
        </button>
      </div>
    </section>
  )
}
