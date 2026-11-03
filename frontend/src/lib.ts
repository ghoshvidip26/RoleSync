import type { ProfileData, ResumeDraft } from './types.ts'

/** "a, b\nc" -> ["a", "b", "c"]: blanks dropped, repeats removed (case-insensitive), order kept. */
export function parseTags(text: string): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const part of text.split(/[,\n;|]/)) {
    const item = part.trim()
    const key = item.toLowerCase()
    if (item && !seen.has(key)) {
      seen.add(key)
      out.push(item)
    }
  }
  return out
}

/** One item per non-blank line. */
export function parseLines(text: string): string[] {
  return text.split('\n').map((l) => l.trim()).filter(Boolean)
}

export const sameList = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i])

/** The backend's stand-in when a resume has no e-mail; it must never be saved as the candidate's address. */
export const PLACEHOLDER_EMAIL = 'candidate@example.com'

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
export const looksLikeEmail = (value: string) => EMAIL.test(value.trim())

/** Problems that stop a profile from being saved, as sentences for the user. */
export function profileProblems(p: Pick<ProfileData, 'name' | 'email'>): string[] {
  const problems: string[] = []
  if (!p.name.trim()) problems.push('Enter your name.')
  if (!looksLikeEmail(p.email) || p.email.trim().toLowerCase() === PLACEHOLDER_EMAIL) problems.push('Enter your real e-mail address.')
  return problems
}

/** A profile as the API wants it: blanks become null, lists are cleaned. */
export function cleanProfile(p: ProfileData): ProfileData {
  const nul = (s: string | null) => (s && s.trim() ? s.trim() : null)
  const list = (xs: string[]) => xs.map((x) => x.trim()).filter(Boolean)
  return {
    name: p.name.trim(), email: p.email.trim(), github: nul(p.github), portfolio: nul(p.portfolio), linkedin: nul(p.linkedin),
    summary: nul(p.summary), skills: list(p.skills), programming_languages: list(p.programming_languages),
    frameworks: list(p.frameworks), databases_tools: list(p.databases_tools), experience: list(p.experience),
    projects: list(p.projects), achievements: list(p.achievements), education: list(p.education), target_roles: p.target_roles,
  }
}

type Editable = Pick<ResumeDraft, 'headline' | 'summary' | 'skills' | 'experience' | 'projects' | 'achievements'>
export const EDITABLE_FIELDS = ['headline', 'summary', 'skills', 'experience', 'projects', 'achievements'] as const

/** Only the draft fields the user actually changed (the API leaves unsent fields alone). */
export function draftChanges(original: ResumeDraft, edited: ResumeDraft): Partial<Editable> {
  const changes: Partial<Editable> = {}
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
  for (const field of EDITABLE_FIELDS) {
    if (!same(original[field], edited[field])) (changes as Record<string, unknown>)[field] = edited[field]
  }
  return changes
}

/** Seconds as "1m 05s". */
export function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`
}

export const errorText = (err: unknown) => (err instanceof Error ? err.message : 'Something went wrong.')

export function problemsIn(d: ResumeDraft): string[] {
  const out: string[] = []
  d.experience.forEach((e, i) => { if (!e.company.trim() || !e.job_title.trim()) out.push(`Role ${i + 1} needs a company and a job title.`) })
  d.projects.forEach((p, i) => { if (!p.name.trim()) out.push(`Project ${i + 1} needs a name.`) })
  if ((d.headline ?? '').length > 120) out.push('The headline can be at most 120 characters.')
  if ((d.summary ?? '').length > 600) out.push('The summary can be at most 600 characters.')
  return out
}
