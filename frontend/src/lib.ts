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

/** Questions a job posting asks the applicant: sentences or lines that end in "?" (3 to 200 characters), at most `limit`. */
export function extractQuestions(text: string, limit = 4): string[] {
  const found: string[] = []
  for (const piece of (text || '').split(/(?<=[.!?])\s+|\n+/)) {
    const q = piece.replace(/^[\s\-*•\d.)]+/, '').replace(/\s+/g, ' ').trim()
    if (q.endsWith('?') && q.length >= 3 && q.length <= 200 && !found.some((x) => x.toLowerCase() === q.toLowerCase())) found.push(q)
  }
  return found.slice(0, limit)
}

/** Words of an e-mail body before its sign-off ("Thanks & Regards,"), which is not counted. */
export function bodyWords(body: string): number {
  const at = body.search(/^[ \t]*(?:thanks[ \t]*(?:&|and)[ \t]*regards|best regards|kind regards|warm regards|regards|sincerely|best|thanks|thank you|cheers)[ \t]*,?[ \t]*$/im)
  return (at >= 0 ? body.slice(0, at) : body).split(/\s+/).filter((w) => w && w !== '•').length   // bullet markers are not words
}

/** One "Unclear" requirement from the fit check, in plain words. */
export interface UnclearItem {
  requirement: string
  /** What the resume does show (words of the requirement it contains), without grammar filler. */
  found: string[]
  /** Words of the requirement the resume does not contain. */
  missing: string[]
  /** The closest line of the resume, when the check found one. */
  closest: string | null
  /** Plain-language explanation of why it is unclear and what to do. */
  note: string
  /** "skill": names tools or technologies; "work": describes the job's duties, nothing exact to look for in a resume. */
  kind: 'skill' | 'work'
}

// Grammar and generic duty words: they say nothing about a skill, so they are not listed as "found" or "missing".
const FILLER = new Set(('a an the and or of for to in on at by with from into over under up out off as is are be been it its this that these those our your their we you '
  + 'carry define shape closely work working worked build building maintain design develop develops implement ensure participate collaborate support help '
  + 'strong ability timely manner way across including such well new existing various related relevant good best great').split(' '))
const meaningful = (words: string[]) => words.map((w) => w.trim()).filter((w) => w && !FILLER.has(w.toLowerCase()))
const splitWords = (list: string | undefined) => (list ? list.split(/,\s*/) : [])
const clip = (text: string, limit = 240) => {
  const t = text.replace(/\s+/g, ' ').trim()
  return t.length <= limit ? t : `${t.slice(0, limit).replace(/\s+\S*$/, '').replace(/[,;:(]+$/, '')}…`
}
// A requirement names something technical if it has an acronym, a digit, or a name written like a tool (GraphQL, Node.js, C++).
const TOOL_SHAPE = /\b[A-Z]{2,}\b|\d|[a-z][A-Z]|[A-Za-z][.+#][A-Za-z+#]/                 // case-sensitive: ML, GraphQL, Node.js, C++
const TOOL_WORD = /\b(?:python|java|react|node|sql|docker|kubernetes|aws|api|apis|rest|git|linux|swift|kotlin|flutter|django|flask|redis|kafka)\b/i
const NAMES_A_TOOL = { test: (text: string) => TOOL_SHAPE.test(text) || TOOL_WORD.test(text) }

/** Turn the fit check's wording ("<requirement> — only partly shown (found: a, b; not found: c)") into something a person can act on.
 * Text in a format it does not know is shown as it came, never dropped. */
export function explainUnclear(raw: string): UnclearItem {
  const at = raw.indexOf(' — ')
  const requirement = (at >= 0 ? raw.slice(0, at) : raw).replace(/[.\s]+$/, '').trim()
  const rest = at >= 0 ? raw.slice(at + 3).trim() : ''
  const kind: UnclearItem['kind'] = NAMES_A_TOOL.test(requirement) ? 'skill' : 'work'
  const base = { requirement, found: [] as string[], missing: [] as string[], closest: null as string | null, kind }

  const partly = rest.match(/^only partly shown \(found: (.*?); (?:not found: (.*)|but never together in one place)\)$/)
  if (partly) {
    const found = meaningful(splitWords(partly[1]))
    const missing = meaningful(splitWords(partly[2]))
    const together = !partly[2]
    const note = together
      ? 'Your resume mentions each of these, but never together in one place, so it is not clear you did this.'
      : kind === 'skill'
        ? 'Your resume has some of these words but not all of them. If you have really used them, add them in your own words. If not, leave it.'
        : 'This describes the work itself rather than one tool, so there is nothing exact to look for. If you have done something like it, say so in your own words.'
    return { ...base, found, missing, note }
  }
  const related = rest.match(/^related evidence exists \((.*)\) but it does not clearly meet the requirement$/)
  if (related) {
    return { ...base, closest: clip(related[1]), note: 'Your resume has related work (below), but it does not clearly show this requirement. Reword that line if it really covers it.' }
  }
  const years = rest.match(/^relevant work is shown \((.*)\), but the required amount of experience cannot be confirmed from the profile$/)
  if (years) {
    return { ...base, closest: clip(years[1]), note: 'Your resume shows relevant work, but the amount of experience the job asks for cannot be confirmed from it.' }
  }
  if (/^cannot be verified from the profile or resume text$/.test(rest)) {
    return { ...base, note: 'This cannot be checked from resume text, so it is neither a plus nor a minus.' }
  }
  if (/^evidence cited but its wording does not match/.test(rest)) {
    return { ...base, note: 'Something in your resume may be related, but its wording is too different to count as a match.' }
  }
  if (/^the stored profile is too sparse/.test(rest)) {
    return { ...base, note: 'Your saved profile has too little detail to check this. Add more to your experience or projects.' }
  }
  return { ...base, note: rest }
}
