import { describe, expect, it } from 'vitest'
import { bodyWords, explainUnclear, cleanProfile, extractQuestions, clock, draftChanges, looksLikeEmail, parseLines, parseTags, PLACEHOLDER_EMAIL, profileProblems, sameList } from './lib.ts'
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

// ---- skills: only skills the profile already has can be highlighted ------------------------------------------------

import { findProfileSkill, matchingProfileSkills, profileSkillList, recommendSkills, sameSkills, skillKey } from './lib.ts'

const PROFILE_SKILLS = {
  skills: ['OOP', 'Python, JavaScript', 'React Native'],
  programming_languages: ['Python', 'TypeScript'],
  frameworks: ['React.js', 'Next.js'],
  databases_tools: ['PostgreSQL', 'postgresql', 'Docker'],
}

describe('skill helpers', () => {
  it('treats spelling variants as one skill', () => {
    expect(skillKey('React Native')).toBe(skillKey('react-native'))
    expect(skillKey('C++')).toBe('c++')
    expect(skillKey('C#')).not.toBe(skillKey('C'))
    expect(skillKey('  ')).toBe('')
  })

  it('lists every profile skill once, splitting comma-joined entries; tools and languages before general skills', () => {
    expect(profileSkillList(PROFILE_SKILLS)).toEqual(['Python', 'TypeScript', 'React.js', 'Next.js', 'PostgreSQL', 'Docker', 'OOP', 'JavaScript', 'React Native'])
    expect(profileSkillList(null)).toEqual([])
    expect(profileSkillList({ skills: [], programming_languages: [], frameworks: [], databases_tools: [] })).toEqual([])
  })

  it("finds the profile's own spelling, and nothing the profile does not have", () => {
    const list = profileSkillList(PROFILE_SKILLS)
    expect(findProfileSkill('react native', list)).toBe('React Native')
    expect(findProfileSkill('REACT-NATIVE', list)).toBe('React Native')
    expect(findProfileSkill('Kubernetes', list)).toBeNull()
    expect(findProfileSkill('', list)).toBeNull()
    expect(findProfileSkill('React', list)).toBeNull() // a part of a skill is not the skill
  })

  it('offers matching profile skills that are not highlighted yet, prefix matches first', () => {
    const list = profileSkillList(PROFILE_SKILLS)
    expect(matchingProfileSkills('re', list, [])).toEqual(['React.js', 'React Native', 'PostgreSQL']) // prefix matches first, then "contains"
    expect(matchingProfileSkills('script', list, [])).toEqual(['TypeScript', 'JavaScript'])
    expect(matchingProfileSkills('re', list, ['React Native'])).toEqual(['React.js', 'PostgreSQL'])
    expect(matchingProfileSkills('zzz', list, [])).toEqual([])
    expect(matchingProfileSkills('', list, [])).toEqual([])
    expect(matchingProfileSkills('a', list, [], 2)).toHaveLength(2)
  })

  it('recommends unhighlighted profile skills, bringing back what the AI had suggested first', () => {
    const list = profileSkillList(PROFILE_SKILLS)
    expect(recommendSkills(list, ['Python'], ['Docker', 'Python'], 3)).toEqual(['Docker', 'TypeScript', 'React.js'])
    expect(recommendSkills(list, list, [])).toEqual([])
    expect(recommendSkills(list, [], [], 2)).toEqual(['Python', 'TypeScript'])
  })

  it('compares skill lists by order and spelling-insensitive value', () => {
    expect(sameSkills(['React Native', 'Go'], ['react-native', 'go'])).toBe(true)
    expect(sameSkills(['Go', 'Rust'], ['Rust', 'Go'])).toBe(false)
    expect(sameSkills(['Go'], [])).toBe(false)
  })
})

describe('extractQuestions / bodyWords', () => {
  it('finds the questions a posting asks, once each, cleaned', () => {
    const jd = 'We build things.\n- Why do you want to work here?\n2) Why do you want to work here?\nTell us. What is your proudest project? Thanks.'
    expect(extractQuestions(jd)).toEqual(['Why do you want to work here?', 'What is your proudest project?'])
    expect(extractQuestions('')).toEqual([])
  })
  it('counts the body words before the sign-off only', () => {
    expect(bodyWords('Hi Sam,\n\nOne two three.\n\nThanks & Regards,\nAlex Doe')).toBe(5)
    expect(bodyWords('No sign off here')).toBe(4)
    expect(bodyWords('Proof:\n• one two\n• three')).toBe(4)          // the • markers are not words
  })
})

describe('explainUnclear', () => {
  it('says which words the resume mentions and which it does not, without filler words', () => {
    const u = explainUnclear('Build scalable and secure REST and GraphQL APIs — only partly shown (found: secure, rest, apis; not found: scalable, graphql)')
    expect(u.requirement).toBe('Build scalable and secure REST and GraphQL APIs')
    expect(u.found).toEqual(['secure', 'rest', 'apis'])
    expect(u.missing).toEqual(['scalable', 'graphql'])
    expect(u.kind).toBe('skill')
    expect(u.note).toMatch(/If you have really used them/)
  })
  it('drops grammar and generic duty words from the lists', () => {
    const u = explainUnclear('Design evaluation framework and carry out evaluations of our query processing engine. — only partly shown (found: evaluations, framework, engine; not found: carry, out, query, processing)')
    expect(u.missing).toEqual(['query', 'processing'])
    expect(u.requirement.endsWith('.')).toBe(false)
  })
  it('treats a duty with no named tool as work, not as a skill to prove', () => {
    const u = explainUnclear('Shape up the engineering culture and tech stack for the company. — only partly shown (found: engineering; not found: shape, up, culture, tech, stack, company)')
    expect(u.kind).toBe('work')
    expect(u.note).toMatch(/describes the work itself/)
  })
  it('explains words that appear but never together', () => {
    const u = explainUnclear('Collaborate with frontend developers to integrate APIs — only partly shown (found: frontend, integrate, apis; but never together in one place)')
    expect(u.note).toMatch(/never together in one place/)
    expect(u.missing).toEqual([])
  })
  it('shows the closest line of the resume for related evidence', () => {
    const u = explainUnclear('Design and optimize PostgreSQL database schemas — related evidence exists (SDE Intern at TechKareer: Optimized PostgreSQL schemas) but it does not clearly meet the requirement')
    expect(u.closest).toBe('SDE Intern at TechKareer: Optimized PostgreSQL schemas')
    expect(u.note).toMatch(/related work/)
  })
  it('explains an experience-amount requirement', () => {
    const u = explainUnclear('2+ years of Python — relevant work is shown (Backend work at Acme), but the required amount of experience cannot be confirmed from the profile')
    expect(u.closest).toBe('Backend work at Acme')
    expect(u.note).toMatch(/amount of experience/)
  })
  it('never drops text in a format it does not know', () => {
    expect(explainUnclear('Some requirement — brand new wording').note).toBe('brand new wording')
    expect(explainUnclear('Just a requirement').requirement).toBe('Just a requirement')
  })
})
