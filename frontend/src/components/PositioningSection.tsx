import { useState, type KeyboardEvent, type ReactNode } from 'react'
import { Pencil, Plus, RefreshCw, Sparkles, Undo2, X } from 'lucide-react'
import { suggestField, type PositioningField } from '../api.ts'
import { useFlow } from '../flow.tsx'
import type { PositioningBaseline } from '../flowModel.ts'
import { errorText, findProfileSkill, matchingProfileSkills, recommendSkills, sameSkills, skillKey } from '../lib.ts'
import { ErrorBanner, Field, TextArea, TextInput } from '../ui.tsx'

export const HEADLINE_MAX = 120
export const SUMMARY_MAX = 600
export const MAX_HIGHLIGHTED = 12

export interface PositioningValues {
  headline: string
  summary: string
  skills: string[]
}

interface Props {
  draftId: string
  values: PositioningValues
  onChange: (patch: Partial<PositioningValues>) => void
  /** Every skill the candidate's profile has. Only these can be highlighted. */
  profileSkills: string[]
  /** The job title the user gave, if any: the headline is written for that role (when the resume supports it). */
  jobTitle: string
  /** Approved drafts are read-only. */
  locked: boolean
}

const LABEL: Record<PositioningField, string> = { headline: 'headline', summary: 'summary', skills: 'highlighted skills' }

/** Who wrote what: the AI's latest suggestion for a field, or the user's own edit of it. */
export function Source({ edited }: { edited: boolean }) {
  return edited ? (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400">
      <Pencil size={11} aria-hidden /> Edited by you
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-indigo-950/50 border border-indigo-800/40 text-indigo-300">
      <Sparkles size={11} aria-hidden /> AI suggestion
    </span>
  )
}

export function PositioningSection({ draftId, values, onChange, profileSkills, jobTitle, locked }: Props) {
  const { flow, update } = useFlow()
  // What the AI last suggested for this draft. A draft from before this existed counts what it holds as the suggestion.
  const saved = flow.positioning?.draftId === draftId ? flow.positioning : null
  const [baseline, setBaseline] = useState<PositioningBaseline>(
    () => saved ?? { draftId, headline: values.headline, summary: values.summary, skills: values.skills, variants: { headline: 0, summary: 0, skills: 0 } },
  )
  const [busy, setBusy] = useState<PositioningField | null>(null)
  const [error, setError] = useState<string | null>(null)

  const remember = (next: PositioningBaseline) => {
    setBaseline(next)
    update({ positioning: next })
  }

  const edited = {
    headline: values.headline.trim() !== baseline.headline.trim(),
    summary: values.summary.trim() !== baseline.summary.trim(),
    skills: !sameSkills(values.skills, baseline.skills),
  }

  const regenerate = async (field: PositioningField) => {
    if (edited[field] && !window.confirm(`Replace your edits to the ${LABEL[field]} with a new suggestion?`)) return
    setBusy(field)
    setError(null)
    try {
      const variant = baseline.variants[field] + 1
      const out = await suggestField(draftId, field, variant, jobTitle)
      const value = out.value
      if (field === 'skills') {
        const list = Array.isArray(value) ? value : []
        if (!list.length) throw new Error('None of your profile skills match this job, so there is nothing to suggest.')
        onChange({ skills: list })
        remember({ ...baseline, skills: list, variants: { ...baseline.variants, skills: variant } })
      } else {
        const text = typeof value === 'string' ? value.trim() : ''
        if (!text) throw new Error(`There is no ${field} to suggest from your resume for this job.`)
        onChange({ [field]: text })
        remember({ ...baseline, [field]: text, variants: { ...baseline.variants, [field]: variant } })
      }
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  const restore = (field: PositioningField) => {
    onChange({ [field]: baseline[field] } as Partial<PositioningValues>)
  }

  const actions = (field: PositioningField) => (
    <>
      <Source edited={edited[field]} />
      {edited[field] && (
        <button type="button" className="btn small ghost" disabled={locked || busy !== null} onClick={() => restore(field)}>
          <Undo2 size={13} aria-hidden /> Restore suggestion
        </button>
      )}
      <button type="button" className="btn small" disabled={locked || busy !== null} onClick={() => void regenerate(field)}
        aria-label={`Regenerate ${LABEL[field]}`}>
        <RefreshCw size={13} aria-hidden className={busy === field ? 'animate-spin' : ''} /> {busy === field ? 'Working…' : 'Regenerate'}
      </button>
    </>
  )

  return (
    <div className="space-y-5" aria-label="Resume positioning">
      <div>
        <h3 className="text-base font-semibold text-white">Resume positioning</h3>
        <p className="muted small mt-1">
          Suggested from your resume and this job. Every word comes from your own profile. Edit anything; regenerating one field leaves the others alone.
        </p>
      </div>
      <ErrorBanner message={error} />

      <TextInput
        label="Headline"
        value={values.headline}
        onChange={(v) => onChange({ headline: v })}
        maxLength={HEADLINE_MAX}
        disabled={locked}
        actions={actions('headline')}
        hint={`${values.headline.length}/${HEADLINE_MAX} · a short line for the role you are applying for`}
      />

      <TextArea
        label="Professional summary"
        value={values.summary}
        onChange={(v) => onChange({ summary: v })}
        rows={3}
        maxLength={SUMMARY_MAX}
        autoResize
        disabled={locked}
        actions={actions('summary')}
      />

      <SkillChips
        skills={values.skills}
        suggested={baseline.skills}
        profileSkills={profileSkills}
        onChange={(skills) => onChange({ skills })}
        locked={locked}
        actions={actions('skills')}
      />
    </div>
  )
}

function SkillChips({ skills, suggested, profileSkills, onChange, locked, actions }: {
  skills: string[]
  suggested: string[]
  profileSkills: string[]
  onChange: (skills: string[]) => void
  locked: boolean
  actions: ReactNode
}) {
  const [text, setText] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const matches = matchingProfileSkills(text, profileSkills, skills)
  const offers = recommendSkills(profileSkills, skills, suggested)
  const atLimit = skills.length >= MAX_HIGHLIGHTED

  const add = (raw: string) => {
    const wanted = raw.trim().replace(/,$/, '').trim()
    if (!wanted) return
    const own = findProfileSkill(wanted, profileSkills)
    if (!own) {
      setProblem(`“${wanted}” is not in your profile, so it cannot be highlighted. Add it to your profile first.`)
      return
    }
    if (skills.some((s) => skillKey(s) === skillKey(own))) {
      setProblem(`${own} is already highlighted.`)
      return
    }
    if (atLimit) {
      setProblem(`Highlight up to ${MAX_HIGHLIGHTED} skills so the best ones stand out.`)
      return
    }
    setProblem(null)
    setText('')
    onChange([...skills, own])
  }

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      // Enter takes the first matching profile skill when what was typed is only part of one
      add(findProfileSkill(text, profileSkills) ? text : (matches[0] ?? text))
    } else if (e.key === 'Backspace' && !text && skills.length && !locked) {
      onChange(skills.slice(0, -1))
    }
  }

  return (
    <Field
      label="Skills to highlight"
      actions={actions}
      hint="Highlighted skills move to the front of their own group on your resume. All your skills stay; none are ever removed."
    >
      {(id) => (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/50 p-2.5 focus-within:border-indigo-500" role="list" aria-label="Highlighted skills">
            {skills.length === 0 && <span className="muted small px-1">No skills highlighted yet.</span>}
            {skills.map((skill) => (
              <span key={skillKey(skill)} role="listitem"
                className="inline-flex items-center gap-1 text-xs font-medium pl-2.5 pr-1 py-1 rounded-md bg-indigo-950/40 border border-indigo-800/40 text-indigo-300">
                {skill}
                <button type="button" aria-label={`Remove ${skill}`} disabled={locked}
                  className="rounded p-0.5 text-indigo-300 hover:bg-indigo-900/60 hover:text-white disabled:opacity-40"
                  onClick={() => onChange(skills.filter((s) => skillKey(s) !== skillKey(skill)))}>
                  <X size={12} aria-hidden />
                </button>
              </span>
            ))}
            <input
              id={id}
              aria-label="Add a skill"
              className="min-w-[10rem] flex-1"
              style={{ width: 'auto', border: 0, background: 'transparent', boxShadow: 'none', padding: '4px' }}
              placeholder={atLimit ? `Up to ${MAX_HIGHLIGHTED} skills` : 'Add a skill from your profile…'}
              value={text}
              disabled={locked}
              onChange={(e) => { setText(e.target.value); setProblem(null) }}
              onKeyDown={onKey}
            />
          </div>

          {problem && <p className="error-text" role="alert">{problem}</p>}

          {text.trim() && matches.length > 0 && (
            <div className="flex flex-wrap gap-2" aria-label="Matching profile skills">
              {matches.map((m) => (
                <button key={m} type="button" className="btn small" disabled={locked || atLimit} onClick={() => add(m)}>
                  <Plus size={12} aria-hidden /> {m}
                </button>
              ))}
            </div>
          )}

          {offers.length > 0 && !text.trim() && (
            <div className="space-y-1.5">
              <span className="muted small">From your profile, click to highlight:</span>
              <div className="flex flex-wrap gap-2">
                {offers.map((o) => (
                  <button key={o} type="button" disabled={locked || atLimit} onClick={() => add(o)}
                    aria-label={`Highlight ${o}`}
                    className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-md border border-slate-700 text-slate-300 hover:border-indigo-500 hover:text-white disabled:opacity-40">
                    <Plus size={11} aria-hidden /> {o}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Field>
  )
}
