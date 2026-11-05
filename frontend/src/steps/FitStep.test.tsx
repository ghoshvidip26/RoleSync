import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api.ts'
import { FlowProvider } from '../flow.tsx'
import { EMPTY_FLOW, FLOW_KEY } from '../flowModel.ts'
import { FitStep } from './FitStep.tsx'

vi.mock('../api.ts', async (original) => ({ ...(await original<typeof import('../api.ts')>()), reviewFit: vi.fn() }))

const JD = 'Build scalable REST and GraphQL APIs. Shape up the engineering culture.'
const REVIEW = {
  verdict: {
    fit_assessment: {
      verdict: 'partial_match', reasoning: 'Partial match.', matching_requirements: [], gaps: [],
      uncertain_requirements: [
        'Build scalable REST and GraphQL APIs — only partly shown (found: rest, apis; not found: scalable, graphql)',
        'Shape up the engineering culture. — only partly shown (found: engineering; not found: shape, up, culture)',
      ],
    },
  },
  tailoring: { needed: true, reasons: ['Add the missing skills.'] },
}

beforeEach(() => {
  sessionStorage.clear()
  vi.clearAllMocks()
  sessionStorage.setItem(FLOW_KEY, JSON.stringify({ ...EMPTY_FLOW, step: 'fit', profile: { id: 'p1', name: 'A', email: 'a@b.co' }, job: { title: '', company: '', link: '', description: JD } }))
  vi.mocked(api.reviewFit).mockResolvedValue(REVIEW as never)
})

async function check() {
  render(<FlowProvider><FitStep onNext={vi.fn()} /></FlowProvider>)
  await userEvent.click(screen.getByRole('button', { name: /Check my fit/ }))
  await screen.findByText(/Partly shown \(unclear\)/)
}

describe('the fit page', () => {
  it('explains each unclear requirement in plain words instead of showing the matcher’s wording', async () => {
    await check()
    expect(screen.getByText(/not a verdict that you lack the skill/)).toBeInTheDocument()
    expect(screen.getByText('Build scalable REST and GraphQL APIs')).toBeInTheDocument()
    expect(screen.getAllByText('Not mentioned:')[0].parentElement?.textContent).toContain('scalable, graphql')
    expect(screen.getAllByText('Your resume mentions:')[0].parentElement?.textContent).toContain('rest, apis')
    expect(screen.queryByText(/only partly shown/)).toBeNull()
    expect(screen.getByText(/describes the work itself/)).toBeInTheDocument()          // the culture sentence is a duty, not a skill
  })

  it('shows only what comes from this job: no built-in role templates', async () => {
    await check()
    expect(screen.queryByText(/roles your profile suits/i)).toBeNull()
    expect(api.reviewFit).toHaveBeenCalledOnce()
  })
})
