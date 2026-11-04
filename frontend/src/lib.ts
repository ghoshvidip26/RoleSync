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

// ---- skills: only skills the candidate's profile already has can be highlighted ----------------------------------

/** "React Native", "react-native" and "React.Native" are one skill. */
export const skillKey = (skill: string) => skill.toLowerCase().replace(/[^a-z0-9+#]/g, '')

/** Every skill in the profile, once each: languages, frameworks and tools first, then the general skills list (which also
 * holds concepts such as "OOP"), so the most useful ones lead when skills are offered. */
export function profileSkillList(
  profile: Pick<ProfileData, 'skills' | 'programming_languages' | 'frameworks' | 'databases_tools'> | null,
): string[] {
  if (!profile) return []
  const seen = new Set<string>()
  const out: string[] = []
  for (const field of [profile.programming_languages, profile.frameworks, profile.databases_tools, profile.skills]) {
    for (const entry of field ?? []) {
      for (const part of entry.split(',')) {
        const skill = part.trim()
        const key = skillKey(skill)
        if (key && !seen.has(key)) {
          seen.add(key)
          out.push(skill)
        }
      }
    }
  }
  return out
}

/** The profile's own spelling of what was typed, or null when the profile has no such skill. */
export const findProfileSkill = (text: string, list: string[]): string | null => {
  const key = skillKey(text)
  return key ? (list.find((s) => skillKey(s) === key) ?? null) : null
}

/** Profile skills that start with, or contain, what was typed (best first). */
export function matchingProfileSkills(text: string, list: string[], taken: string[], limit = 6): string[] {
  const key = skillKey(text)
  if (!key) return []
  const used = new Set(taken.map(skillKey))
  const open = list.filter((s) => !used.has(skillKey(s)) && skillKey(s).includes(key))
  return [...open.filter((s) => skillKey(s).startsWith(key)), ...open.filter((s) => !skillKey(s).startsWith(key))].slice(0, limit)
}

/** Skills to offer as one-click additions: the profile's skills that are not highlighted yet, those the AI had
 * suggested first (so a removed suggestion is easy to bring back), then the rest in profile order. */
export function recommendSkills(list: string[], selected: string[], suggested: string[], limit = 12): string[] {
  const used = new Set(selected.map(skillKey))
  const open = list.filter((s) => !used.has(skillKey(s)))
  const wanted = new Set(suggested.map(skillKey))
  return [...open.filter((s) => wanted.has(skillKey(s))), ...open.filter((s) => !wanted.has(skillKey(s)))].slice(0, limit)
}

/** Same skills in the same order, ignoring spelling differences. */
export const sameSkills = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((s, i) => skillKey(s) === skillKey(b[i]))
