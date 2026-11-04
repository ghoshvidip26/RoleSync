import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api.ts'
import { FlowProvider } from '../flow.tsx'
import { EMPTY_FLOW, FLOW_KEY } from '../flowModel.ts'
import { PositioningSection, type PositioningValues } from './PositioningSection.tsx'

vi.mock('../api.ts', async (original) => ({ ...(await original<typeof import('../api.ts')>()), suggestField: vi.fn() }))

const PROFILE_SKILLS = ['Python', 'JavaScript', 'TypeScript', 'React Native', 'Firebase', 'PostgreSQL', 'Docker', 'FAISS']
const AI: PositioningValues = {
  headline: 'React Native Developer | JavaScript, Firebase',
  summary: 'Software engineer with 1 year of experience shipping mobile and AI features. Delivered 6 rider order flows.',
  skills: ['React Native', 'JavaScript', 'Firebase'],
}

/** The section needs its parent's state (the draft editor holds it), so a tiny parent plays that role. */
function Harness({ initial = AI, locked = false, jobTitle = '', onChange }: { initial?: PositioningValues; locked?: boolean; jobTitle?: string; onChange?: (v: PositioningValues) => void }) {
  const [values, setValues] = useState(initial)
  return (
    <FlowProvider>
      <PositioningSection draftId="d1" values={values} profileSkills={PROFILE_SKILLS} jobTitle={jobTitle} locked={locked}
        onChange={(patch) => setValues((v) => { const next = { ...v, ...patch }; onChange?.(next); return next })} />
      <output data-testid="skills">{values.skills.join('|')}</output>
      <output data-testid="headline">{values.headline}</output>
      <output data-testid="summary">{values.summary}</output>
    </FlowProvider>
  )
}

const field = (name: string) => screen.getByLabelText(name) as HTMLInputElement | HTMLTextAreaElement
const skills = () => screen.getByTestId('skills').textContent
const row = (label: string) => screen.getByText(label, { selector: 'label' }).closest('.field') as HTMLElement
const confirmWith = (answer: boolean) => vi.spyOn(window, 'confirm').mockReturnValue(answer)

beforeEach(() => {
  sessionStorage.clear()
  vi.mocked(api.suggestField).mockReset()
  vi.restoreAllMocks()
})

describe('AI suggestion vs your own edit', () => {
  it('shows every field as an AI suggestion at first', () => {
    render(<Harness />)
    expect(screen.getAllByText('AI suggestion')).toHaveLength(3)
    expect(screen.queryByText('Edited by you')).toBeNull()
  })

  it('marks only the field the user changed, and restores the suggestion', async () => {
    render(<Harness />)
    await userEvent.type(field('Headline'), ' 2')
    expect(within(row('Headline')).getByText('Edited by you')).toBeInTheDocument()
    expect(within(row('Professional summary')).getByText('AI suggestion')).toBeInTheDocument()
    expect(within(row('Skills to highlight')).getByText('AI suggestion')).toBeInTheDocument()
    await userEvent.click(within(row('Headline')).getByRole('button', { name: /Restore suggestion/ }))
    expect(screen.getByTestId('headline')).toHaveTextContent(AI.headline)
    expect(screen.queryByText('Edited by you')).toBeNull()
  })

  it('changing the skills marks the skills as edited', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Remove Firebase' }))
    expect(within(row('Skills to highlight')).getByText('Edited by you')).toBeInTheDocument()
    await userEvent.click(within(row('Skills to highlight')).getByRole('button', { name: /Restore suggestion/ }))
    expect(skills()).toBe('React Native|JavaScript|Firebase')
  })
})

describe('regenerating one field', () => {
  it('replaces only that field, keeps the others as they are, and sends the job title', async () => {
    vi.mocked(api.suggestField).mockResolvedValue({ field: 'headline', variant: 1, value: 'JavaScript, Firebase | React Native Developer' })
    render(<Harness jobTitle="React Native Developer" />)
    await userEvent.type(field('Professional summary'), ' My own words.')            // the user edited the summary
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    await waitFor(() => expect(screen.getByTestId('headline')).toHaveTextContent('JavaScript, Firebase | React Native Developer'))
    expect(api.suggestField).toHaveBeenCalledWith('d1', 'headline', 1, 'React Native Developer')
    expect(screen.getByTestId('summary')).toHaveTextContent('My own words.')              // untouched
    expect(skills()).toBe('React Native|JavaScript|Firebase')
    expect(within(row('Headline')).getByText('AI suggestion')).toBeInTheDocument()       // the new text is the suggestion now
    expect(within(row('Professional summary')).getByText('Edited by you')).toBeInTheDocument()
  })

  it('asks for a different suggestion each time', async () => {
    vi.mocked(api.suggestField).mockResolvedValueOnce({ field: 'summary', variant: 1, value: 'First new summary.' })
      .mockResolvedValueOnce({ field: 'summary', variant: 2, value: 'Second new summary.' })
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate summary' }))
    await waitFor(() => expect(screen.getByTestId('summary')).toHaveTextContent('First new summary.'))
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate summary' }))
    await waitFor(() => expect(screen.getByTestId('summary')).toHaveTextContent('Second new summary.'))
    expect(vi.mocked(api.suggestField).mock.calls.map((c) => c[2])).toEqual([1, 2])
  })

  it("asks before throwing away the user's own edits, and keeps them if declined", async () => {
    render(<Harness />)
    await userEvent.type(field('Headline'), '!')
    const ask = confirmWith(false)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    expect(ask).toHaveBeenCalledOnce()
    expect(api.suggestField).not.toHaveBeenCalled()
    expect(screen.getByTestId('headline')).toHaveTextContent(`${AI.headline}!`)
  })

  it('replaces the edits when the user agrees', async () => {
    vi.mocked(api.suggestField).mockResolvedValue({ field: 'headline', variant: 1, value: 'New headline' })
    render(<Harness />)
    await userEvent.type(field('Headline'), '!')
    confirmWith(true)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    await waitFor(() => expect(screen.getByTestId('headline')).toHaveTextContent('New headline'))
  })

  it('regenerates the skills from the profile only', async () => {
    vi.mocked(api.suggestField).mockResolvedValue({ field: 'skills', variant: 1, value: ['Python', 'Docker'] })
    render(<Harness />)
    confirmWith(true)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate highlighted skills' }))
    await waitFor(() => expect(skills()).toBe('Python|Docker'))
  })

  it('shows the reason and changes nothing when there is no suggestion or the call fails', async () => {
    vi.mocked(api.suggestField).mockResolvedValueOnce({ field: 'skills', variant: 1, value: [] })
    render(<Harness />)
    confirmWith(true)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate highlighted skills' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('None of your profile skills match this job')
    expect(skills()).toBe('React Native|JavaScript|Firebase')
    vi.mocked(api.suggestField).mockRejectedValueOnce(new api.ApiError('Only drafts pending review can get new suggestions', 409))
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    expect(await screen.findByText(/Only drafts pending review/)).toBeInTheDocument()
    expect(screen.getByTestId('headline')).toHaveTextContent(AI.headline)
  })

  it('shows progress while one field is working and blocks the others', async () => {
    let finish: (v: { field: 'headline'; variant: number; value: string }) => void = () => undefined
    vi.mocked(api.suggestField).mockReturnValue(new Promise((resolve) => { finish = resolve }))
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    expect(screen.getByRole('button', { name: 'Regenerate headline' })).toHaveTextContent('Working…')
    expect(screen.getByRole('button', { name: 'Regenerate summary' })).toBeDisabled()
    await act(async () => finish({ field: 'headline', variant: 1, value: 'Done' }))
    expect(screen.getByRole('button', { name: 'Regenerate summary' })).toBeEnabled()
  })

  it('remembers the suggestion across leaving and returning to the page', async () => {
    vi.mocked(api.suggestField).mockResolvedValue({ field: 'headline', variant: 1, value: 'Remembered headline' })
    const { unmount } = render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate headline' }))
    await waitFor(() => expect(screen.getByTestId('headline')).toHaveTextContent('Remembered headline'))
    unmount()
    render(<Harness initial={{ ...AI, headline: 'Remembered headline' }} />)      // the page reopens with the saved text
    expect(within(row('Headline')).getByText('AI suggestion')).toBeInTheDocument()
    expect(JSON.parse(sessionStorage.getItem(FLOW_KEY) ?? '{}').positioning.headline).toBe('Remembered headline')
  })
})

describe('the summary box', () => {
  it('keeps the character limit and counter', async () => {
    render(<Harness />)
    const box = field('Professional summary') as HTMLTextAreaElement
    expect(box.maxLength).toBe(600)
    expect(row('Professional summary')).toHaveTextContent(`${AI.summary.length}/600`)
    await userEvent.type(box, 'abc')
    expect(row('Professional summary')).toHaveTextContent(`${AI.summary.length + 3}/600`)
  })

  it('grows and shrinks with its text', async () => {
    let content = 60                                                         // what the browser would measure for the current text
    const spy = vi.spyOn(HTMLTextAreaElement.prototype, 'scrollHeight', 'get').mockImplementation(() => content)
    render(<Harness initial={{ ...AI, summary: 'Short summary text here' }} />)
    const box = field('Professional summary')
    expect(box.style.height).toBe('62px')                                   // measured on first show
    content = 140
    await userEvent.type(box, '.')
    expect(box.style.height).toBe('142px')                                  // grew with the text
    content = 60
    await userEvent.type(box, '.')
    expect(box.style.height).toBe('62px')                                   // and shrank again
    expect(box.style.overflow).toBe('hidden')
    spy.mockRestore()
  })

  it('the headline has its own 120 character limit and counter', () => {
    render(<Harness />)
    expect((field('Headline') as HTMLInputElement).maxLength).toBe(120)
    expect(row('Headline')).toHaveTextContent(`${AI.headline.length}/120`)
  })
})

describe('highlighted skills', () => {
  it('shows chips, removes one, and never offers a way to remove a profile skill', async () => {
    render(<Harness />)
    expect(screen.getAllByRole('listitem').map((c) => c.textContent)).toEqual(['React Native', 'JavaScript', 'Firebase'])
    await userEvent.click(screen.getByRole('button', { name: 'Remove JavaScript' }))
    expect(skills()).toBe('React Native|Firebase')
    expect(screen.queryByRole('button', { name: /Delete|Remove (Python|Docker|FAISS)/ })).toBeNull()
  })

  it('explains that highlighting only reorders and never removes', () => {
    render(<Harness />)
    expect(screen.getByText(/move to the front of their own group on your resume\. All your skills stay; none are ever removed/)).toBeInTheDocument()
  })

  it("adds a profile skill typed in any spelling, using the profile's own", async () => {
    render(<Harness />)
    await userEvent.type(screen.getByLabelText('Add a skill'), 'type-script{Enter}')
    expect(skills()).toBe('React Native|JavaScript|Firebase|TypeScript')
    await userEvent.type(screen.getByLabelText('Add a skill'), 'docker,')
    expect(skills()).toBe('React Native|JavaScript|Firebase|TypeScript|Docker')
  })

  it('completes a part of a profile skill on Enter', async () => {
    render(<Harness />)
    await userEvent.type(screen.getByLabelText('Add a skill'), 'postg{Enter}')
    expect(skills()).toBe('React Native|JavaScript|Firebase|PostgreSQL')
  })

  it('refuses a skill the profile does not have and says why', async () => {
    const changed = vi.fn()
    render(<Harness onChange={changed} />)
    await userEvent.type(screen.getByLabelText('Add a skill'), 'Kubernetes{Enter}')
    expect(screen.getByRole('alert')).toHaveTextContent('“Kubernetes” is not in your profile, so it cannot be highlighted. Add it to your profile first.')
    expect(skills()).toBe('React Native|JavaScript|Firebase')
    expect(changed).not.toHaveBeenCalled()
    await userEvent.type(screen.getByLabelText('Add a skill'), 'x')                    // typing again clears the message
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('does not add a skill twice', async () => {
    render(<Harness />)
    await userEvent.type(screen.getByLabelText('Add a skill'), 'firebase{Enter}')
    expect(screen.getByRole('alert')).toHaveTextContent('Firebase is already highlighted.')
    expect(skills()).toBe('React Native|JavaScript|Firebase')
  })

  it('suggests matching profile skills while typing and adds one on click', async () => {
    render(<Harness />)
    await userEvent.type(screen.getByLabelText('Add a skill'), 'scr')
    const matches = within(screen.getByLabelText('Matching profile skills'))
    expect(matches.getAllByRole('button').map((b) => b.textContent?.trim())).toEqual(['TypeScript'])
    await userEvent.click(matches.getByRole('button', { name: /TypeScript/ }))
    expect(skills()).toBe('React Native|JavaScript|Firebase|TypeScript')
  })

  it('recommends profile skills that are not highlighted, and adds one on click', async () => {
    render(<Harness />)
    const offered = screen.getAllByRole('button', { name: /^Highlight / }).map((b) => b.getAttribute('aria-label'))
    expect(offered).toEqual(['Highlight Python', 'Highlight TypeScript', 'Highlight PostgreSQL', 'Highlight Docker', 'Highlight FAISS'])
    await userEvent.click(screen.getByRole('button', { name: 'Highlight Docker' }))
    expect(skills()).toBe('React Native|JavaScript|Firebase|Docker')
    expect(screen.queryByRole('button', { name: 'Highlight Docker' })).toBeNull()
  })

  it('brings back a removed AI suggestion first', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Remove Firebase' }))
    expect(screen.getAllByRole('button', { name: /^Highlight / })[0]).toHaveAccessibleName('Highlight Firebase')
  })

  it('removes the last chip with Backspace on an empty box', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByLabelText('Add a skill'))
    await userEvent.keyboard('{Backspace}')
    expect(skills()).toBe('React Native|JavaScript')
  })

  it('stops at 12 highlighted skills', async () => {
    const many = Array.from({ length: 8 }, (_, i) => `Skill${i}`)
    const six = ['Python', 'JavaScript', 'TypeScript', 'React Native', 'Firebase', 'PostgreSQL']
    render(<FlowProvider><LimitHarness profile={[...six, 'Docker', 'FAISS', ...many]} start={[...six, ...many.slice(0, 6)]} /></FlowProvider>)
    expect(screen.getByLabelText('Add a skill')).toHaveAttribute('placeholder', 'Up to 12 skills')
    await userEvent.type(screen.getByLabelText('Add a skill'), 'Docker{Enter}')
    expect(screen.getByRole('alert')).toHaveTextContent('Highlight up to 12 skills')
    expect(screen.getAllByRole('listitem')).toHaveLength(12)
  })
})

function LimitHarness({ profile, start }: { profile: string[]; start: string[] }) {
  const [values, setValues] = useState<PositioningValues>({ ...AI, skills: start })
  return <PositioningSection draftId="d1" values={values} profileSkills={profile} jobTitle="" locked={false} onChange={(p) => setValues((v) => ({ ...v, ...p }))} />
}

describe('an approved (locked) draft', () => {
  it('cannot be changed or regenerated', async () => {
    render(<Harness locked />)
    expect(field('Headline')).toBeDisabled()
    expect(field('Professional summary')).toBeDisabled()
    expect(screen.getByLabelText('Add a skill')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Remove Firebase' })).toBeDisabled()
    for (const name of ['Regenerate headline', 'Regenerate summary', 'Regenerate highlighted skills']) expect(screen.getByRole('button', { name })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Highlight Docker' })).toBeDisabled()
  })
})

describe('without any flow state', () => {
  it('starts from the values it is given', () => {
    sessionStorage.setItem(FLOW_KEY, JSON.stringify(EMPTY_FLOW))
    render(<Harness />)
    expect(screen.getAllByText('AI suggestion')).toHaveLength(3)
  })
})
