import { ArrowRight, Briefcase, Building2, Sparkles, Target } from 'lucide-react'
import { useFlow } from '../flow.tsx'
import { MIN_JOB_CHARS } from '../steps.ts'
import { StepHeader, TextArea, TextInput } from '../ui.tsx'

const MAX_JOB_CHARS = 20_000

export function JobStep({ onNext, onSkipFit }: { onNext: () => void; onSkipFit: () => void }) {
  const { flow, update } = useFlow()
  const job = flow.job
  const set = (changes: Partial<typeof job>) => update({ job: { ...job, ...changes } })
  const length = job.description.trim().length
  const ready = length >= MIN_JOB_CHARS
  const charProgress = Math.min(100, Math.round((length / MIN_JOB_CHARS) * 100))

  return (
    <section className="space-y-6">
      <StepHeader
        title="The job"
        intro="Paste the job description. We check how well you fit, then tailor your resume to it."
      />

      {/* Target Company & Title */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs tracking-wider uppercase">
          <Briefcase size={14} /> Target Role Details
        </div>
        <div className="grid2">
          <TextInput
            label="Job title (optional)"
            value={job.title}
            onChange={(v) => set({ title: v })}
            hint="e.g. Senior Frontend Engineer"
          />
          <TextInput
            label="Company (optional)"
            value={job.company}
            onChange={(v) => set({ company: v })}
            hint="Used to fill in your e-mail later."
          />
        </div>
      </div>

      {/* Job Description Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs tracking-wider uppercase">
            <Building2 size={14} /> Full Job Posting
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className={ready ? 'text-emerald-400 font-medium' : 'text-slate-400'}>
              {length} / {MIN_JOB_CHARS} min chars
            </span>
            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  ready ? 'bg-emerald-400' : 'bg-indigo-500'
                }`}
                style={{ width: `${charProgress}%` }}
              />
            </div>
          </div>
        </div>

        <TextArea
          label="Job description"
          value={job.description}
          onChange={(v) => set({ description: v })}
          rows={12}
          maxLength={MAX_JOB_CHARS}
          hint={ready ? undefined : `Paste at least ${MIN_JOB_CHARS} characters`}
        />
      </div>

      <div className="actions">
        <button className="btn primary" disabled={!ready} onClick={onNext}>
          <Target size={16} />
          Check my fit
        </button>
        <button className="btn" disabled={!ready} onClick={onSkipFit}>
          <Sparkles size={15} />
          Skip to tailoring
          <ArrowRight size={14} />
        </button>
      </div>
    </section>
  )
}
