import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api.ts'
import { FlowProvider } from '../flow.tsx'
import { EMPTY_FLOW, FLOW_KEY } from '../flowModel.ts'
import type { Gallery, PdfResult } from '../types.ts'
import { DesignStep } from './DesignStep.tsx'

vi.mock('../api.ts', async (original) => ({ ...(await original<typeof import('../api.ts')>()), getGallery: vi.fn(), resumePdf: vi.fn() }))

const TEMPLATES = ['classic', 'modern', 'harvard', 'compact'].map((id) => ({
  id, name: id[0].toUpperCase() + id.slice(1), description: '', best_for: `best for ${id}`, ats_notes: '', sections: [], preview_url: `/x/${id}`,
}))
const own = (over = {}) => ({ id: 'original' as const, name: 'Your uploaded design', available: true, preserve_supported: true, reason: null, ats: null, ...over })
const gallery = (original: Gallery['original']): Gallery => ({ templates: TEMPLATES, original })
const pdf = (renderer: string): PdfResult => ({ url: 'blob:x', renderer, validation: 'pass', pages: 1 })

const state = (over = {}) => sessionStorage.setItem(FLOW_KEY, JSON.stringify({
  ...EMPTY_FLOW, step: 'design', profile: { id: 'p1', name: 'Ann', email: 'a@b.co' },
  draft: { id: 'd1', profile_id: 'p1', status: 'approved', draft: {} }, ...over,
}))
const saved = () => JSON.parse(sessionStorage.getItem(FLOW_KEY) ?? '{}')
const show = (onNext = vi.fn()) => { render(<FlowProvider><DesignStep onNext={onNext} /></FlowProvider>); return onNext }

beforeEach(() => {
  sessionStorage.clear()
  vi.mocked(api.getGallery).mockReset()
  vi.mocked(api.resumePdf).mockReset()
})

describe('keeping the resume the user uploaded', () => {
  it("builds the user's own design automatically and lets them continue with it straight away", async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    vi.mocked(api.resumePdf).mockResolvedValue(pdf('original'))
    const onNext = show()
    const go = await screen.findByRole('button', { name: "Continue with my resume's design" })
    expect(api.resumePdf).toHaveBeenCalledWith('d1', 'original')
    expect(saved().template).toBe('original')                           // chosen without any extra click
    await userEvent.click(go)
    expect(onNext).toHaveBeenCalledOnce()
  })

  it('labels the own design as the default and the card button as "Keep my design"', async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    vi.mocked(api.resumePdf).mockResolvedValue(pdf('original'))
    show()
    expect(await screen.findByText('default')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Keep my design' })).toBeEnabled()
  })

  it('can go back to the own design after looking at a template', async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    vi.mocked(api.resumePdf).mockImplementation(async (_id, template) => pdf(template === 'original' ? 'original' : `template:${template}`))
    show()
    await screen.findByRole('button', { name: "Continue with my resume's design" })
    await userEvent.click(screen.getAllByRole('button', { name: 'Preview with my resume' })[0])
    expect(await screen.findByRole('button', { name: 'Use this design' })).toBeInTheDocument()
    expect(saved().template).toBe('classic')
    await userEvent.click(screen.getByRole('button', { name: 'Keep my design' }))
    expect(await screen.findByRole('button', { name: "Continue with my resume's design" })).toBeInTheDocument()
    expect(saved().template).toBe('original')
  })

  it('explains why the own design is unavailable, builds nothing by itself, and offers the templates', async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own({ preserve_supported: false, reason: 'Only single-column resumes can be rebuilt' })))
    show()
    expect(await screen.findByText(/Not available: Only single-column resumes can be rebuilt/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Keep my design' })).toBeDisabled()
    expect(api.resumePdf).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: /Continue|Use this design/ })).toBeNull()
    await userEvent.click(screen.getAllByRole('button', { name: 'Preview with my resume' })[1])
    expect(api.resumePdf).toHaveBeenCalledWith('d1', 'modern')
  })

  it('when no resume PDF was stored it says so and does not break', async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own({ available: false, preserve_supported: false, reason: null })))
    show()
    expect(await screen.findByText(/no resume PDF was stored/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Keep my design' })).toBeDisabled()
  })

  it('a failed build shows the reason, keeps nothing chosen, and the templates stay usable', async () => {
    state()
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    vi.mocked(api.resumePdf).mockRejectedValueOnce(
      new api.ChooseTemplateError('Your resume\'s own design could not be reproduced reliably for this draft.', TEMPLATES, [{ name: 'text_complete', detail: '1 text is missing' }], null))
    show()
    expect(await screen.findByRole('alert')).toHaveTextContent('could not be reproduced reliably')
    expect(screen.getByText(/text_complete: 1 text is missing/)).toBeInTheDocument()
    expect(saved().template).toBeNull()
    expect(screen.queryByRole('button', { name: /Continue|Use this design/ })).toBeNull()
    vi.mocked(api.resumePdf).mockResolvedValueOnce(pdf('template:classic'))
    await userEvent.click(screen.getAllByRole('button', { name: 'Preview with my resume' })[0])
    expect(await screen.findByRole('button', { name: 'Use this design' })).toBeEnabled()
  })

  it('reopens the design that was chosen before instead of asking again', async () => {
    state({ template: 'harvard' })
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    vi.mocked(api.resumePdf).mockResolvedValue(pdf('template:harvard'))
    show()
    await waitFor(() => expect(api.resumePdf).toHaveBeenCalledWith('d1', 'harvard'))
    expect(await screen.findByRole('button', { name: 'Use this design' })).toBeInTheDocument()
  })

  it('asks for approval first when the draft is not approved', async () => {
    state({ draft: { id: 'd1', profile_id: 'p1', status: 'pending_review', draft: {} } })
    vi.mocked(api.getGallery).mockResolvedValue(gallery(own()))
    show()
    expect(await screen.findByText('Approve your tailored resume first.')).toBeInTheDocument()
    expect(api.resumePdf).not.toHaveBeenCalled()
  })
})
