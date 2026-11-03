import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import App from './App.tsx'
import { EMPTY_FLOW, FLOW_KEY, type FlowState } from './flowModel.ts'
import { lockReason } from './steps.ts'
import type { DraftRecord, Profile } from './types.ts'

const profile = { id: 'p1', name: 'Ann', email: 'ann@example.com' } as Profile
const draft = (status: DraftRecord['status']) => ({ id: 'd1', profile_id: 'p1', status, draft: {} }) as DraftRecord
const flow = (c: Partial<FlowState>): FlowState => ({ ...EMPTY_FLOW, ...c })
const JD = { title: '', company: '', link: '', description: 'x'.repeat(60) }
const save = (state: Partial<FlowState>) => sessionStorage.setItem(FLOW_KEY, JSON.stringify({ ...EMPTY_FLOW, ...state }))

describe('which steps are open', () => {
  it('opens only the first step on a fresh start', () => {
    const f = flow({})
    expect(lockReason('upload', f)).toBeNull()
    for (const s of ['job', 'fit', 'draft', 'design', 'email'] as const) expect(lockReason(s, f)).toBeTruthy()
  })
  it('opens the job step once there is a profile, and fit/tailor once there is a job description', () => {
    expect(lockReason('job', flow({ profile }))).toBeNull()
    expect(lockReason('fit', flow({ profile }))).toMatch(/job description/)
    expect(lockReason('fit', flow({ profile, job: JD }))).toBeNull()
    expect(lockReason('draft', flow({ profile, job: JD }))).toBeNull()
  })
  it('needs an approved draft for the design and a chosen design for the e-mail', () => {
    expect(lockReason('design', flow({ profile, job: JD, draft: draft('pending_review') }))).toMatch(/Approve/)
    expect(lockReason('design', flow({ profile, job: JD, draft: draft('approved') }))).toBeNull()
    expect(lockReason('email', flow({ profile, job: JD, draft: draft('approved') }))).toMatch(/design/)
    expect(lockReason('email', flow({ profile, job: JD, draft: draft('approved'), template: 'classic' }))).toBeNull()
    expect(lockReason('email', flow({ profile, job: JD, draft: draft('rejected'), template: 'classic' }))).toMatch(/Approve/)
  })
})

describe('the app', () => {
  beforeEach(() => sessionStorage.clear())

  it('starts on the upload step with the later steps disabled', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Upload your resume' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Resume/ })).toBeEnabled()
    for (const name of [/Job/, /Fit/, /Tailor/, /Design/, /E-mail/]) expect(screen.getByRole('button', { name })).toBeDisabled()
  })

  it('refuses a file that is not a PDF', async () => {
    render(<App />)
    await userEvent.upload(screen.getByLabelText('Resume PDF'), new File(['hi'], 'notes.txt', { type: 'text/plain' }), { applyAccept: false })
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose a PDF file.')
    expect(screen.queryByRole('button', { name: 'Read my resume' })).toBeNull()
  })

  it('accepts a PDF and offers to read it', async () => {
    render(<App />)
    await userEvent.upload(screen.getByLabelText('Resume PDF'), new File(['%PDF-1.4'], 'cv.pdf', { type: 'application/pdf' }))
    expect(await screen.findByRole('button', { name: 'Read my resume' })).toBeEnabled()
  })

  it('restores a saved flow after a reload and opens the job step', () => {
    save({ step: 'job', profile })
    render(<App />)
    expect(screen.getByRole('heading', { name: 'The job' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Check my fit' })).toBeDisabled() // no description yet
  })

  it('falls back to the first step when a saved step is no longer allowed', () => {
    save({ step: 'email' })
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Upload your resume' })).toBeInTheDocument()
  })

  it('needs at least 40 characters of job description', async () => {
    save({ step: 'job', profile })
    render(<App />)
    const box = screen.getByLabelText('Job description')
    await userEvent.type(box, 'too short')
    expect(screen.getByRole('button', { name: 'Check my fit' })).toBeDisabled()
    await userEvent.clear(box)
    await userEvent.type(box, 'Build cross-platform mobile apps using React Native and TypeScript.')
    expect(screen.getByRole('button', { name: 'Check my fit' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Skip to tailoring' })).toBeEnabled()
  })
})
