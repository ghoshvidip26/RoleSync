import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, ChooseTemplateError, createOutreach, describeDetail, emailAttachment, extractProfile, resumePdf, sendEmail } from './api.ts'

const reply = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers })

afterEach(() => vi.unstubAllGlobals())
const stub = (fn: (url: string, init?: RequestInit) => Response | Promise<Response>) => {
  const mock = vi.fn(async (url: string, init?: RequestInit) => fn(url, init))
  vi.stubGlobal('fetch', mock)
  return mock
}

describe('describeDetail', () => {
  it('uses a string detail as is', () => expect(describeDetail('Nope', 'x')).toBe('Nope'))
  it('turns FastAPI validation problems into a sentence', () => {
    const detail = [{ loc: ['body', 'questions', 0], msg: 'Value error, each question must be 3 to 200 characters' },
                    { loc: ['body', 'email'], msg: 'value is not a valid email address' }]
    expect(describeDetail(detail, 'x')).toBe('questions: each question must be 3 to 200 characters; email: value is not a valid email address')
  })
  it('falls back when there is nothing usable', () => {
    expect(describeDetail(undefined, 'fallback')).toBe('fallback'); expect(describeDetail('  ', 'fallback')).toBe('fallback'); expect(describeDetail([], 'fallback')).toBe('fallback')
  })
})

describe('errors', () => {
  it('reads the backend message', async () => {
    stub(() => reply(502, { detail: 'Could not extract any requirements.' }))
    await expect(extractProfile(new File(['x'], 'r.pdf'))).rejects.toMatchObject({ message: 'Could not extract any requirements.', status: 502 })
  })
  it('explains an unreadable error body by status', async () => {
    stub(() => reply(500, 'Internal Server Error'))
    await expect(extractProfile(new File(['x'], 'r.pdf'))).rejects.toMatchObject({ message: 'The server hit an unexpected error.', status: 500 })
  })
  it('says so when the server cannot be reached', async () => {
    stub(() => { throw new TypeError('Failed to fetch') })
    await expect(extractProfile(new File(['x'], 'r.pdf'))).rejects.toMatchObject({ message: 'Cannot reach the server. Is the backend running?', status: 0 })
  })
  it('turns the choose-template answer into ChooseTemplateError with the gallery', async () => {
    stub(() => reply(409, { detail: 'No original resume PDF is stored.', action: 'choose_template', templates: [{ id: 'classic' }], failed_checks: [{ name: 'text_complete', detail: 'missing' }] }))
    const err = await resumePdf('d1', null).catch((e) => e)
    expect(err).toBeInstanceOf(ChooseTemplateError)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.templates).toEqual([{ id: 'classic' }]); expect(err.failedChecks).toEqual([{ name: 'text_complete', detail: 'missing' }])
  })
  it('keeps an ordinary 409 as a plain ApiError', async () => {
    stub(() => reply(409, { detail: 'Approve the resume draft before generating its PDF' }))
    const err = await resumePdf('d1', 'classic').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError); expect(err).not.toBeInstanceOf(ChooseTemplateError)
  })
})

describe('requests', () => {
  it('sends the template as a query parameter, encoded, and only when chosen', async () => {
    const mock = stub(() => reply(200, 'pdf', { 'x-resume-renderer': 'template:classic', 'x-resume-validation': 'pass', 'x-resume-pages': '2' }))
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => undefined }))
    const out = await resumePdf('d1', 'classic')
    expect(mock.mock.calls[0][0]).toBe('/api/resume-drafts/d1/pdf?template=classic')
    expect(out).toMatchObject({ url: 'blob:x', renderer: 'template:classic', validation: 'pass', pages: 2 })
    await emailAttachment('e1', null)
    expect(mock.mock.calls[1][0]).toBe('/api/emails/e1/attachment')
    await sendEmail('e1', 'original').catch(() => undefined)
    expect(mock.mock.calls[2][0]).toBe('/api/emails/e1/approve?template=original')
  })
  it('posts outreach as JSON with the approved draft, questions and note', async () => {
    const mock = stub(() => reply(201, { id: 'e1', status: 'pending_review' }))
    await createOutreach({ profile_id: 'p', company_id: 'c', recipient_id: 'r', resume_draft_id: 'd', questions: ['Why me?'], personal_note: null })
    const [url, init] = mock.mock.calls[0]
    expect(url).toBe('/api/outreach'); expect(init?.method).toBe('POST')
    expect(JSON.parse(String(init?.body))).toEqual({ profile_id: 'p', company_id: 'c', recipient_id: 'r', resume_draft_id: 'd', questions: ['Why me?'], personal_note: null })
  })
})
