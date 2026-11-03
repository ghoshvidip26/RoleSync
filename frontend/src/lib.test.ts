import { describe, expect, it } from 'vitest'
import { cleanProfile, clock, draftChanges, looksLikeEmail, parseLines, parseTags, PLACEHOLDER_EMAIL, profileProblems, sameList } from './lib.ts'
import type { ProfileData, ResumeDraft } from './types.ts'

describe('parseTags', () => {
  it('splits on commas, newlines, semicolons and pipes; trims; drops blanks', () => {
    expect(parseTags(' Python , Go;SQL\n|  Docker ,, ')).toEqual(['Python', 'Go', 'SQL', 'Docker'])
  })
  it('removes repeats whatever their case and keeps the first spelling and order', () => {
    expect(parseTags('Go, python, GO, Python')).toEqual(['Go', 'python'])
  })
  it('gives nothing for empty input', () => {
    expect(parseTags('')).toEqual([])
    expect(parseTags(' , ; ')).toEqual([])
  })
})

describe('parseLines / sameList / clock', () => {
  it('keeps one item per non-blank line', () => expect(parseLines('a\n\n  b  \n')).toEqual(['a', 'b']))
  it('compares lists by value and order', () => {
    expect(sameList(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(sameList(['a', 'b'], ['b', 'a'])).toBe(false)
    expect(sameList(['a'], ['a', 'b'])).toBe(false)
  })
  it('formats seconds', () => {
    expect(clock(0)).toBe('0s'); expect(clock(59)).toBe('59s'); expect(clock(65)).toBe('1m 05s'); expect(clock(600)).toBe('10m 00s')
  })
})

describe('profile validation', () => {
  it('accepts normal addresses and rejects bad ones', () => {
    for (const ok of ['a@b.co', ' x.y+z@mail.example.com ']) expect(looksLikeEmail(ok)).toBe(true)
    for (const bad of ['', 'a', 'a@b', '@b.co', 'a b@c.co']) expect(looksLikeEmail(bad)).toBe(false)
  })
  it('never lets the backend placeholder address be saved', () => {
    expect(profileProblems({ name: 'Ann', email: PLACEHOLDER_EMAIL })).toEqual(['Enter your real e-mail address.'])
    expect(profileProblems({ name: ' ', email: 'ann@example.com' })).toEqual(['Enter your name.'])
    expect(profileProblems({ name: 'Ann', email: 'ann@example.com' })).toEqual([])
  })
  it('cleanProfile trims text, turns blanks into null and drops empty list items', () => {
    const p: ProfileData = {
      name: ' Ann ', email: ' ann@example.com ', github: '  ', portfolio: null, linkedin: ' linkedin.com/in/ann ', summary: '',
      skills: [' Go ', ''], programming_languages: [], frameworks: [], databases_tools: [], experience: ['  ', 'Role | Co'],
      projects: [], achievements: [], education: [], target_roles: null,
    }
    const c = cleanProfile(p)
    expect(c).toMatchObject({ name: 'Ann', email: 'ann@example.com', github: null, linkedin: 'linkedin.com/in/ann', summary: null, skills: ['Go'], experience: ['Role | Co'] })
  })
})

const DRAFT: ResumeDraft = {
  headline: 'H', summary: 'S', skills: ['Go'], achievements: ['A'], education: [{ details: 'BSc' }], recommendations: [], claims_requiring_confirmation: [],
  experience: [{ company: 'Acme', job_title: 'Eng', duration: null, location: null, bullets: ['x'] }],
  projects: [{ name: 'P', technologies: ['Go'], description: null, bullets: [] }],
}

describe('draftChanges', () => {
  it('is empty when nothing changed', () => expect(draftChanges(DRAFT, structuredClone(DRAFT))).toEqual({}))
  it('returns only the fields that changed', () => {
    const edited = structuredClone(DRAFT)
    edited.summary = 'New'
    edited.experience[0].bullets.push('y')
    expect(draftChanges(DRAFT, edited)).toEqual({ summary: 'New', experience: edited.experience })
  })
  it('ignores read-only fields such as education and recommendations', () => {
    const edited = structuredClone(DRAFT)
    edited.education = [{ details: 'changed' }]
    edited.recommendations = ['x']
    expect(draftChanges(DRAFT, edited)).toEqual({})
  })
})
