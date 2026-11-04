import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api.ts'
import { FlowProvider } from '../flow.tsx'
import { EMPTY_FLOW, FLOW_KEY } from '../flowModel.ts'
import type { DraftRecord, ResumeDraft } from '../types.ts'
import { DraftStep } from './DraftStep.tsx'

vi.mock('../api.ts', async (original) => ({
  ...(await original<typeof import('../api.ts')>()),
  editDraft: vi.fn(), approveDraft: vi.fn(), rejectDraft: vi.fn(), tailorResume: vi.fn(), suggestField: vi.fn(),
}))

const DRAFT: ResumeDraft = {
  headline: 'React Native Developer | JavaScript, Firebase',
  summary: 'Software engineer with 1 year of experience shipping mobile and AI features.',
  skills: ['React Native', 'JavaScript', 'Firebase'],
  experience: [{ company: 'Aforro', job_title: 'Junior Mobile Application Developer', duration: 'Jul 2026 – Present', location: 'Bengaluru', bullets: ['Delivered 6 rider order flows.'] }],
  projects: [{ name: 'StackSearch', technologies: ['FAISS'], description: null, bullets: ['Achieved sub-100 ms retrieval.'] }],
  achievements: ['Won a hackathon.'], education: [{ details: 'B.Tech' }], recommendations: [], claims_requiring_confirmation: [],
}
const PROFILE = {
  id: 'p1', name: 'Vidip', email: 'v@example.com', github: null, portfolio: null, linkedin: null, summary: null,
  skills: ['React Native', 'Python, JavaScript'], programming_languages: ['TypeScript'], frameworks: ['Next.js'], databases_tools: ['Firebase', 'Docker'],
  experience: [], projects: [], achievements: [], education: [], target_roles: null,
}
const record = (status: DraftRecord['status'] = 'pending_review', draft: ResumeDraft = DRAFT): DraftRecord => ({ id: 'd1', profile_id: 'p1', status, draft })
const BASELINE = { draftId: 'd1', headline: DRAFT.headline, summary: DRAFT.summary, skills: DRAFT.skills, variants: { headline: 0, summary: 0, skills: 0 } }

const open = (over: Record<string, unknown> = {}) => {
  sessionStorage.setItem(FLOW_KEY, JSON.stringify({
    ...EMPTY_FLOW, step: 'draft', profile: PROFILE, draft: record(), positioning: BASELINE,
    job: { title: 'React Native Developer', company: '', link: '', description: 'x'.repeat(80) }, ...over,
  }))
  const onNext = vi.fn()
  render(<FlowProvider><DraftStep onNext={onNext} /></FlowProvider>)
  return onNext
}
const saved = () => JSON.parse(sessionStorage.getItem(FLOW_KEY) ?? '{}')
const box = (name: string) => screen.getByLabelText(name) as HTMLInputElement | HTMLTextAreaElement
const SAVE = () => screen.getByRole('button', { name: /Save changes/ })

beforeEach(() => {
  sessionStorage.clear()
  for (const fn of [api.editDraft, api.approveDraft, api.rejectDraft, api.tailorResume, api.suggestField]) vi.mocked(fn).mockReset()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

describe('saving edits to the positioning section', () => {
  it('shows the section with the draft values and nothing to save yet', () => {
    open()
    expect(box('Headline').value).toBe(DRAFT.headline)
    expect(box('Professional summary').value).toBe(DRAFT.summary)
    expect(screen.getAllByRole('listitem').map((c) => c.textContent)).toEqual(['React Native', 'JavaScript', 'Firebase'])
    expect(SAVE()).toBeDisabled()
    expect(screen.queryByText('Unsaved changes')).toBeNull()
  })

  it('sends only the fields that changed, and the page then shows them as saved', async () => {
    const after = record('pending_review', { ...DRAFT, summary: 'My own summary.' })
    vi.mocked(api.editDraft).mockResolvedValue(after)
    open()
    await userEvent.clear(box('Professional summary'))
    await userEvent.type(box('Professional summary'), 'My own summary.')
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument()
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledOnce())
    expect(api.editDraft).toHaveBeenCalledWith('d1', { summary: 'My own summary.' })          // headline and skills are not re-sent
    await waitFor(() => expect(screen.queryByText('Unsaved changes')).toBeNull())
    expect(saved().draft.draft.summary).toBe('My own summary.')
    expect(SAVE()).toBeDisabled()
  })

  it('sends headline, summary and skills together when all three changed', async () => {
    vi.mocked(api.editDraft).mockResolvedValue(record())
    open()
    await userEvent.type(box('Headline'), ' (mobile)')
    await userEvent.type(box('Professional summary'), ' More.')
    await userEvent.click(screen.getByRole('button', { name: 'Remove Firebase' }))
    await userEvent.type(screen.getByLabelText('Add a skill'), 'docker{Enter}')
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledOnce())
    expect(api.editDraft).toHaveBeenCalledWith('d1', {
      headline: `${DRAFT.headline} (mobile)`, summary: `${DRAFT.summary} More.`, skills: ['React Native', 'JavaScript', 'Docker'],
    })
  })

  it('removing or reordering highlighted skills is saved; the profile itself is never part of the request', async () => {
    vi.mocked(api.editDraft).mockResolvedValue(record())
    open()
    await userEvent.click(screen.getByRole('button', { name: 'Remove JavaScript' }))
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledWith('d1', { skills: ['React Native', 'Firebase'] }))
    const sent = vi.mocked(api.editDraft).mock.calls[0][1]
    expect(Object.keys(sent)).toEqual(['skills'])
  })

  it('a cleared headline or summary is sent as empty (null on the server) rather than skipped', async () => {
    vi.mocked(api.editDraft).mockResolvedValue(record())
    open()
    await userEvent.clear(box('Headline'))
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledWith('d1', { headline: null }))
  })

  it('shows the error and keeps the edits when the save is refused', async () => {
    vi.mocked(api.editDraft).mockRejectedValue(new api.ApiError('Only drafts pending review can be edited (current: approved)', 409))
    open()
    await userEvent.type(box('Headline'), '!')
    await userEvent.click(SAVE())
    expect(await screen.findByText(/Only drafts pending review can be edited/)).toBeInTheDocument()
    expect(box('Headline').value).toBe(`${DRAFT.headline}!`)
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument()
  })

  it('a regenerated field is saved like any other edit', async () => {
    vi.mocked(api.suggestField).mockResolvedValue({ field: 'headline', variant: 1, value: 'JavaScript, Firebase | React Native Developer' })
    vi.mocked(api.editDraft).mockResolvedValue(record())
    open()
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    await waitFor(() => expect(box('Headline').value).toBe('JavaScript, Firebase | React Native Developer'))
    expect(api.suggestField).toHaveBeenCalledWith('d1', 'headline', 1, 'React Native Developer')
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledWith('d1', { headline: 'JavaScript, Firebase | React Native Developer' }))
  })
})

describe('approving', () => {
  it('saves the edits first, then approves, then locks the section', async () => {
    const calls: string[] = []
    vi.mocked(api.editDraft).mockImplementation(async () => { calls.push('edit'); return record() })
    vi.mocked(api.approveDraft).mockImplementation(async () => { calls.push('approve'); return record('approved') })
    open()
    await userEvent.type(box('Headline'), '!')
    await userEvent.click(screen.getByRole('button', { name: /Save and approve/ }))
    await waitFor(() => expect(screen.getByRole('button', { name: /Choose a design/ })).toBeInTheDocument())
    expect(calls).toEqual(['edit', 'approve'])
    expect(box('Headline')).toBeDisabled()
    expect(screen.getByLabelText('Add a skill')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Regenerate summary' })).toBeDisabled()
  })

  it('approves straight away when nothing changed, with no save request', async () => {
    vi.mocked(api.approveDraft).mockResolvedValue(record('approved'))
    open()
    await userEvent.click(screen.getByRole('button', { name: /^Approve/ }))
    await waitFor(() => expect(api.approveDraft).toHaveBeenCalledWith('d1'))
    expect(api.editDraft).not.toHaveBeenCalled()
  })

  it('does not approve when the save fails', async () => {
    vi.mocked(api.editDraft).mockRejectedValue(new api.ApiError('headline: String should have at most 120 characters', 422))
    open()
    await userEvent.type(box('Headline'), '!')
    await userEvent.click(screen.getByRole('button', { name: /Save and approve/ }))
    expect(await screen.findByText(/at most 120 characters/)).toBeInTheDocument()
    expect(api.approveDraft).not.toHaveBeenCalled()
  })

  it('an already approved draft shows the section read-only', () => {
    open({ draft: record('approved') })
    expect(box('Headline')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Regenerate headline' })).toBeDisabled()
    expect(screen.getByRole('button', { name: /Choose a design/ })).toBeInTheDocument()
  })
})

describe('the first tailoring', () => {
  it('sends the job title the user gave, and records what the AI wrote as the baseline', async () => {
    vi.mocked(api.tailorResume).mockResolvedValue({ draft_id: 'd9', status: 'pending_review', draft: DRAFT })
    open({ draft: null, positioning: null })
    await userEvent.click(screen.getByRole('button', { name: /Tailor my resume/ }))
    await waitFor(() => expect(api.tailorResume).toHaveBeenCalledWith('p1', 'x'.repeat(80), 'React Native Developer'))
    expect(await screen.findByLabelText('Headline')).toBeInTheDocument()
    expect(saved().positioning).toEqual({ draftId: 'd9', headline: DRAFT.headline, summary: DRAFT.summary, skills: DRAFT.skills, variants: { headline: 0, summary: 0, skills: 0 } })
    expect(within(screen.getByLabelText('Resume positioning')).getAllByText('AI suggestion')).toHaveLength(3)
  })

  it('works without a job title', async () => {
    vi.mocked(api.tailorResume).mockResolvedValue({ draft_id: 'd9', status: 'pending_review', draft: DRAFT })
    open({ draft: null, positioning: null, job: { title: '', company: '', link: '', description: 'y'.repeat(60) } })
    await userEvent.click(screen.getByRole('button', { name: /Tailor my resume/ }))
    await waitFor(() => expect(api.tailorResume).toHaveBeenCalledWith('p1', 'y'.repeat(60), ''))
  })
})

describe('the rest of the editor is unaffected', () => {
  it('still shows experience, projects and achievements next to the new section', () => {
    open()
    expect(screen.getByDisplayValue('Junior Mobile Application Developer')).toBeInTheDocument()
    expect(screen.getByDisplayValue('StackSearch')).toBeInTheDocument()
    expect(screen.getByLabelText('Achievements 1')).toHaveValue('Won a hackathon.')
    expect(screen.getByText(/Education is kept exactly as in your resume/)).toBeInTheDocument()
  })

  it('editing an experience bullet is still saved on its own', async () => {
    vi.mocked(api.editDraft).mockResolvedValue(record())
    open()
    const bullets = screen.getAllByLabelText('Bullets')[0] as HTMLTextAreaElement
    await userEvent.type(bullets, '{Enter}Shipped a feature.')
    await userEvent.click(SAVE())
    await waitFor(() => expect(api.editDraft).toHaveBeenCalledOnce())
    expect(Object.keys(vi.mocked(api.editDraft).mock.calls[0][1])).toEqual(['experience'])
  })
})
