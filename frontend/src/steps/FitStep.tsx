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
import { recommendRoles, reviewFit } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { errorText } from '../lib.ts'
import type { RoleRecommendation } from '../types.ts'
import { Badge, Collapsible, ErrorBanner, Notice, StepHeader, Working } from '../ui.tsx'

const fitTone = (fit: string) =>
  (fit.startsWith('strong') ? 'good' : fit.startsWith('partial') ? 'warn' : 'bad') as
    | 'good'
    | 'warn'
    | 'bad'
const label = (fit: string) => fit.replace(/_/g, ' ')

function Roles({ roles }: { roles: RoleRecommendation[] }) {
  return (
    <div className="cards">
      {roles.slice(0, 4).map((r) => {
        const pct = Math.round(r.score * 100)
        return (
          <article className="card" key={r.role_id}>
            <div className="flex items-start justify-between gap-2">
              <h4 className="m-0 text-white font-semibold">{r.title}</h4>
              <Badge tone={fitTone(r.fit)}>{label(r.fit)}</Badge>
            </div>

            {/* Score progress bar */}
            <div className="space-y-1 my-1">
              <div className="flex justify-between text-xs">
                <span className="muted">Match alignment</span>
                <span className="font-mono font-medium text-indigo-400">{pct}%</span>
              </div>
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-sky-400 rounded-full"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            {r.matching.required.length > 0 && (
              <p className="small text-slate-300">
                <strong className="text-emerald-400">You have:</strong>{' '}
                {[...r.matching.required, ...r.matching.preferred].join(', ')}
              </p>
            )}
            {(r.gaps.required.length > 0 || r.gaps.preferred.length > 0) && (
              <p className="small text-slate-300">
                <strong className="text-amber-400">Missing:</strong>{' '}
                {[...r.gaps.required, ...r.gaps.preferred].join(', ')}
              </p>
            )}
          </article>
        )
      })}
    </div>
  )
}

export function FitStep({ onNext }: { onNext: () => void }) {
  const { flow, update } = useFlow()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rolesError, setRolesError] = useState<string | null>(null)
  const description = flow.job.description.trim()
  const fit = flow.fit && flow.fit.jobDescription === description ? flow.fit : null // results for another text are stale

  const run = async () => {
    if (!flow.profile) return
    setBusy(true)
    setError(null)
    setRolesError(null)
    const [roles, review] = await Promise.allSettled([
      recommendRoles(flow.profile.id),
      reviewFit(flow.profile.id, description),
    ])
    if (roles.status === 'rejected') setRolesError(errorText(roles.reason))
    if (review.status === 'rejected') setError(errorText(review.reason))
    update({
      fit: {
        jobDescription: description,
        roles: roles.status === 'fulfilled' ? roles.value.recommendations : null,
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
      <ErrorBanner message={rolesError} />

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
              <Collapsible title="Unclear" count={assessment.uncertain_requirements.length}>
                <ul className="space-y-2 mt-2">
                  {assessment.uncertain_requirements.map((g, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                      <HelpCircle size={16} className="text-slate-400 shrink-0 mt-0.5" />
                      <span>{g}</span>
                    </li>
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

      {fit?.roles && fit.roles.length > 0 && (
        <div className="space-y-3 pt-2">
          <h3 className="text-base font-semibold text-white">Roles that suit your profile</h3>
          <Roles roles={fit.roles} />
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
