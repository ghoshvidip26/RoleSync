import { useState } from 'react'
import { recommendRoles, reviewFit } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { errorText } from '../lib.ts'
import type { RoleRecommendation } from '../types.ts'
import { Badge, Collapsible, ErrorBanner, Notice, StepHeader, Working } from '../ui.tsx'

const fitTone = (fit: string) => (fit.startsWith('strong') ? 'good' : fit.startsWith('partial') ? 'warn' : 'bad') as 'good' | 'warn' | 'bad'
const label = (fit: string) => fit.replace(/_/g, ' ')

function Roles({ roles }: { roles: RoleRecommendation[] }) {
  return (
    <div className="cards">
      {roles.slice(0, 4).map((r) => (
        <article className="card" key={r.role_id}>
          <h4>{r.title} <Badge tone={fitTone(r.fit)}>{label(r.fit)}</Badge></h4>
          <p className="small muted">Score {Math.round(r.score * 100)}%</p>
          {r.matching.required.length > 0 && <p className="small"><strong>You have:</strong> {[...r.matching.required, ...r.matching.preferred].join(', ')}</p>}
          {(r.gaps.required.length > 0 || r.gaps.preferred.length > 0) && (
            <p className="small"><strong>Missing:</strong> {[...r.gaps.required, ...r.gaps.preferred].join(', ')}</p>
          )}
        </article>
      ))}
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
    const [roles, review] = await Promise.allSettled([recommendRoles(flow.profile.id), reviewFit(flow.profile.id, description)])
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
    <section>
      <StepHeader title="How well do you fit?" intro="We compare your profile with the job and say what is covered, what is missing and what is unclear." />
      {!fit && !busy && !error && (
        <div className="actions"><button className="btn primary" onClick={run}>Check my fit</button></div>
      )}
      {busy && <Working label="Comparing you with the job" hint="This runs on a local model and usually takes one to two minutes. You can stay on this page." />}
      <ErrorBanner message={error} onRetry={run} />
      <ErrorBanner message={rolesError} />

      {assessment && (
        <>
          <h3>This job <Badge tone={fitTone(assessment.verdict)}>{label(assessment.verdict)}</Badge></h3>
          <p>{assessment.reasoning}</p>
          <Collapsible title="Covered by your resume" count={assessment.matching_requirements.length} open>
            <ul>{assessment.matching_requirements.map((m, i) => <li key={i}>{m.split(' — ')[0]}</li>)}</ul>
          </Collapsible>
          {assessment.gaps.length > 0 && <Collapsible title="Not shown in your resume" count={assessment.gaps.length} open><ul>{assessment.gaps.map((g, i) => <li key={i}>{g}</li>)}</ul></Collapsible>}
          {assessment.uncertain_requirements.length > 0 && (
            <Collapsible title="Unclear" count={assessment.uncertain_requirements.length}><ul>{assessment.uncertain_requirements.map((g, i) => <li key={i}>{g}</li>)}</ul></Collapsible>
          )}
          {fit?.review && (
            <Notice tone={fit.review.tailoring.needed ? 'warn' : 'good'}>
              {fit.review.tailoring.needed
                ? <>Tailoring will help. <span className="small">{fit.review.tailoring.reasons.slice(0, 2).join(' ')}</span></>
                : 'Your resume already suits this job. You can still tailor it.'}
            </Notice>
          )}
        </>
      )}
      {fit?.roles && fit.roles.length > 0 && (<><h3>Roles that suit your profile</h3><Roles roles={fit.roles} /></>)}

      {fit && !busy && (
        <div className="actions">
          <button className="btn primary" onClick={onNext}>Tailor my resume</button>
          <button className="btn" onClick={run}>Check again</button>
        </div>
      )}
    </section>
  )
}
