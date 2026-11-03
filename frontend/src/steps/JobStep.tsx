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

  return (
    <section>
      <StepHeader title="The job" intro="Paste the job description. We check how well you fit, then tailor your resume to it." />
      <div className="grid2">
        <TextInput label="Job title (optional)" value={job.title} onChange={(v) => set({ title: v })} />
        <TextInput label="Company (optional)" value={job.company} onChange={(v) => set({ company: v })} hint="Used to fill in your e-mail later." />
      </div>
      <TextArea label="Job description" value={job.description} onChange={(v) => set({ description: v })} rows={14} maxLength={MAX_JOB_CHARS}
        hint={ready ? undefined : `Paste at least ${MIN_JOB_CHARS} characters`} />
      <div className="actions">
        <button className="btn primary" disabled={!ready} onClick={onNext}>Check my fit</button>
        <button className="btn" disabled={!ready} onClick={onSkipFit}>Skip to tailoring</button>
      </div>
    </section>
  )
}
