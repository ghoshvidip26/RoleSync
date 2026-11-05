import { useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  HelpCircle,
  RefreshCw,
  Target,
  Wand2,
} from 'lucide-react'
import { reviewFit } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { errorText, explainUnclear } from '../lib.ts'
import { Badge, Collapsible, ErrorBanner, Notice, StepHeader, Working } from '../ui.tsx'

const fitTone = (fit: string) =>
  (fit.startsWith('strong') ? 'good' : fit.startsWith('partial') ? 'warn' : 'bad') as
    | 'good'
    | 'warn'
    | 'bad'
const label = (fit: string) => fit.replace(/_/g, ' ')

function UnclearCard({ item }: { item: ReturnType<typeof explainUnclear> }) {
  return (
    <li className="rounded-lg border border-slate-800 bg-slate-950/40 p-3.5 space-y-1.5 list-none">
      <div className="flex items-start gap-2">
        <HelpCircle size={16} className="text-slate-400 shrink-0 mt-0.5" aria-hidden />
        <div className="space-y-1.5 text-sm text-slate-300">
          <p className="m-0 font-medium text-white">{item.requirement}</p>
          <p className="m-0">{item.note}</p>
          {item.found.length > 0 && (
            <p className="m-0 small">
              <strong className="text-emerald-400">Your resume mentions:</strong> {item.found.join(', ')}
            </p>
          )}
          {item.missing.length > 0 && (
            <p className="m-0 small">
              <strong className="text-amber-400">Not mentioned:</strong> {item.missing.join(', ')}
            </p>
          )}
          {item.closest && (
            <p className="m-0 small">
              <strong className="text-sky-300">Closest line in your resume:</strong> {item.closest}
            </p>
          )}
        </div>
      </div>
    </li>
  )
}

export function FitStep({ onNext }: { onNext: () => void }) {
  const { flow, update } = useFlow()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const description = flow.job.description.trim()
  const fit = flow.fit && flow.fit.jobDescription === description ? flow.fit : null // results for another text are stale

  const run = async () => {
    if (!flow.profile) return
    setBusy(true)
    setError(null)
    const [review] = await Promise.allSettled([reviewFit(flow.profile.id, description)])
    if (review.status === 'rejected') setError(errorText(review.reason))
    update({
      fit: {
        jobDescription: description,
        review: review.status === 'fulfilled' ? review.value : null,
      },
    })
    setBusy(false)
  }

  const assessment = fit?.review?.verdict.fit_assessment

  return (
    <section className="space-y-6">
      <StepHeader
        title="How well do you fit?"
        intro="We compare your profile with the job and say what is covered, what is missing and what is unclear."
      />

      {!fit && !busy && !error && (
        <div className="actions">
          <button className="btn primary" onClick={run}>
            <Target size={16} />
            Check my fit
          </button>
        </div>
      )}

      {busy && (
        <Working
          label="Comparing you with the job"
          hint="This runs on a local model and usually takes one to two minutes. You can stay on this page."
        />
      )}

      <ErrorBanner message={error} onRetry={run} />

      {assessment && (
        <div className="space-y-5">
          {/* Verdict Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="m-0 text-lg font-bold text-white flex items-center gap-2">
                This job <Badge tone={fitTone(assessment.verdict)}>{label(assessment.verdict)}</Badge>
              </h3>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed">{assessment.reasoning}</p>
          </div>

          {/* Breakdown Sections */}
          <div className="space-y-3">
            <Collapsible
              title="Covered by your resume"
              count={assessment.matching_requirements.length}
              open
            >
              <ul className="space-y-2 mt-2">
                {assessment.matching_requirements.map((m, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                    <CheckCircle2 size={16} className="text-emerald-400 shrink-0 mt-0.5" />
                    <span>{m.split(' — ')[0]}</span>
                  </li>
                ))}
              </ul>
            </Collapsible>

            {assessment.gaps.length > 0 && (
              <Collapsible title="Not shown in your resume" count={assessment.gaps.length} open>
                <ul className="space-y-2 mt-2">
                  {assessment.gaps.map((g, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                      <AlertCircle size={16} className="text-amber-400 shrink-0 mt-0.5" />
                      <span>{g}</span>
                    </li>
                  ))}
                </ul>
              </Collapsible>
            )}

            {assessment.uncertain_requirements.length > 0 && (
              <Collapsible title="Partly shown (unclear)" count={assessment.uncertain_requirements.length} open>
                <p className="muted small mt-2 mb-3">
                  Your resume backs these up in part, but not clearly enough to count them as covered. This is not a verdict that you lack
                  the skill: if you really have it, say so in your resume in your own words.
                </p>
                <ul className="space-y-3">
                  {assessment.uncertain_requirements.map((raw, i) => (
                    <UnclearCard key={i} item={explainUnclear(raw)} />
                  ))}
                </ul>
              </Collapsible>
            )}
          </div>

          {fit?.review && (
            <Notice tone={fit.review.tailoring.needed ? 'warn' : 'good'}>
              {fit.review.tailoring.needed ? (
                <>
                  <strong className="text-amber-300">Tailoring will help.</strong>{' '}
                  <span className="small text-slate-300">
                    {fit.review.tailoring.reasons.slice(0, 2).join(' ')}
                  </span>
                </>
              ) : (
                'Your resume already suits this job. You can still tailor it.'
              )}
            </Notice>
          )}
        </div>
      )}

      {fit && !busy && (
        <div className="actions">
          <button className="btn primary" onClick={onNext}>
            <Wand2 size={16} />
            Tailor my resume
            <ArrowRight size={15} />
          </button>
          <button className="btn" onClick={run}>
            <RefreshCw size={14} />
            Check again
          </button>
        </div>
      )}
    </section>
  )
}
