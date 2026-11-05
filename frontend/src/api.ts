import type {
  DraftRecord, EmailRecord, Gallery, OriginalAssessment, PdfResult, Profile, ProfileData, ResumeDraft,
  ReviewResult, TemplateInfo,
  AnswerResult, AttachmentInfo, PersonalLineResult,
} from './types.ts'

/** Every request goes to /api (proxied to FastAPI by Vite). */
export const API_BASE = '/api'

/** Calls that run the local language model can take minutes. */
export const SLOW_MS = 5 * 60_000
const FAST_MS = 30_000

export class ApiError extends Error {
  status: number
  body: unknown
  constructor(message: string, status: number, body: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

/** The user has to pick a template before a PDF can be made (the API answered 409 + action "choose_template"). */
export class ChooseTemplateError extends ApiError {
  templates: TemplateInfo[]
  failedChecks: Array<{ name: string; detail: string }>
  constructor(message: string, templates: TemplateInfo[], failedChecks: Array<{ name: string; detail: string }>, body: unknown) {
    super(message, 409, body)
    this.name = 'ChooseTemplateError'
    this.templates = templates
    this.failedChecks = failedChecks
  }
}

/** FastAPI's `detail` is a string, or a list of validation problems; turn either into one readable sentence. */
export function describeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (!d || typeof d !== 'object') return ''
        const item = d as { loc?: unknown[]; msg?: string }
        const field = (item.loc ?? []).filter((x) => x !== 'body' && typeof x === 'string').join('.')
        const msg = (item.msg ?? '').replace(/^Value error, /, '')
        return field ? `${field}: ${msg}` : msg
      })
      .filter(Boolean)
    if (parts.length) return parts.join('; ')
  }
  return fallback
}

function statusMessage(status: number): string {
  if (status === 404) return 'Not found.'
  if (status === 413) return 'That file is too large.'
  if (status === 415) return 'That file type is not supported.'
  if (status === 502 || status === 503) return 'A service the server depends on is not available right now. Try again.'
  if (status >= 500) return 'The server hit an unexpected error.'
  return `Request failed (${status}).`
}

async function send(path: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<Response> {
  const { timeoutMs = FAST_MS, ...rest } = init
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(`${API_BASE}${path}`, { ...rest, signal: rest.signal ?? controller.signal })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('This is taking too long. The server may be busy; try again.', 0)
    }
    throw new ApiError('Cannot reach the server. Is the backend running?', 0)
  } finally {
    clearTimeout(timer)
  }
}

async function readError(res: Response): Promise<ApiError> {
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* not JSON */
  }
  const b = body as { detail?: unknown; action?: string; templates?: TemplateInfo[]; failed_checks?: Array<{ name: string; detail: string }> } | null
  const message = describeDetail(b?.detail, statusMessage(res.status))
  if (res.status === 409 && b?.action === 'choose_template') {
    return new ChooseTemplateError(message, b.templates ?? [], b.failed_checks ?? [], body)
  }
  return new ApiError(message, res.status, body)
}

async function json<T>(path: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<T> {
  const res = await send(path, init)
  if (!res.ok) throw await readError(res)
  return (await res.json()) as T
}

const post = (body: unknown) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

function pdfForm(file: File): FormData {
  const form = new FormData()
  form.append('file', file)
  return form
}

// ---- profile ---------------------------------------------------------------------------------------------------

export const extractProfile = (file: File) =>
  json<ProfileData & { raw_resume_text?: string | null }>('/profiles/resume', { method: 'POST', body: pdfForm(file), timeoutMs: SLOW_MS })

export const createProfile = (data: ProfileData) => json<Profile>('/profiles', { ...post(data), timeoutMs: SLOW_MS })

export const uploadOriginalResume = (profileId: string, file: File) =>
  json<OriginalAssessment>(`/profiles/${profileId}/original-resume`, { method: 'POST', body: pdfForm(file), timeoutMs: 60_000 })

// ---- job fit and tailoring --------------------------------------------------------------------------------------


export const reviewFit = (profileId: string, jobDescription: string) =>
  json<ReviewResult>(`/profiles/${profileId}/review`, { ...post({ job_description: jobDescription }), timeoutMs: SLOW_MS })

export const tailorResume = (profileId: string, jobDescription: string, jobTitle = '') =>
  json<{ draft_id: string | null; status: string; draft: ResumeDraft }>(`/profiles/${profileId}/tailor-resume`, {
    // the job title is sent only when the user gave one, so the request is unchanged otherwise
    ...post({ job_description: jobDescription, ...(jobTitle.trim() ? { job_title: jobTitle.trim() } : {}) }),
    timeoutMs: SLOW_MS,
  })

/** A new suggestion for one field of the positioning section. Nothing is saved: the page decides what to keep. */
export type PositioningField = 'headline' | 'summary' | 'skills'
export const suggestField = (draftId: string, field: PositioningField, variant: number, jobTitle = '') =>
  json<{ field: PositioningField; variant: number; value: string | string[] | null }>(`/resume-drafts/${draftId}/suggest`, {
    ...post({ field, variant, ...(jobTitle.trim() ? { job_title: jobTitle.trim() } : {}) }),
    timeoutMs: 60_000,
  })

export const getDraft = (draftId: string) => json<DraftRecord>(`/resume-drafts/${draftId}`)
export const editDraft = (draftId: string, changes: Partial<Pick<ResumeDraft, 'headline' | 'summary' | 'skills' | 'experience' | 'projects' | 'achievements'>>) =>
  json<DraftRecord>(`/resume-drafts/${draftId}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(changes) })
export const approveDraft = (draftId: string) => json<DraftRecord>(`/resume-drafts/${draftId}/approve`, { method: 'POST' })
export const rejectDraft = (draftId: string) => json<DraftRecord>(`/resume-drafts/${draftId}/reject`, { method: 'POST' })

// ---- design and PDF ---------------------------------------------------------------------------------------------

export const getGallery = (profileId: string) => json<Gallery>(`/resume-templates?profile_id=${encodeURIComponent(profileId)}`)
export const templatePreviewUrl = (templateId: string) => `${API_BASE}/resume-templates/${encodeURIComponent(templateId)}/preview`

async function pdfFrom(path: string): Promise<PdfResult> {
  const res = await send(path, { timeoutMs: 120_000 })
  if (!res.ok) throw await readError(res)
  const blob = await res.blob()
  return {
    url: URL.createObjectURL(blob),
    renderer: res.headers.get('x-resume-renderer') ?? '',
    validation: res.headers.get('x-resume-validation') ?? '',
    pages: Number(res.headers.get('x-resume-pages') ?? '0') || 0,
  }
}

const withTemplate = (template: string | null) => (template ? `?template=${encodeURIComponent(template)}` : '')
export const resumePdf = (draftId: string, template: string | null) => pdfFrom(`/resume-drafts/${draftId}/pdf${withTemplate(template)}`)
export const emailAttachment = (emailId: string, template: string | null) => pdfFrom(`/emails/${emailId}/attachment${withTemplate(template)}`)

// ---- outreach ---------------------------------------------------------------------------------------------------

export const createCompany = (name: string, website: string | null) =>
  json<{ id: string }>('/companies', post({ name, website }))
export const createRecipient = (name: string | null, email: string, role: string | null) =>
  json<{ id: string }>('/recipients', post({ name, email, role }))
export interface OutreachContext {
  profile_id: string; company_id: string; recipient_id: string; resume_draft_id: string | null
  job_description?: string | null; job_title?: string | null
}
export const createOutreach = (body: OutreachContext & {
  questions?: string[]; personal_note: string | null; answers?: Array<{ question: string; answer: string }>
}) => json<EmailRecord>('/outreach', { ...post(body), timeoutMs: SLOW_MS })

/** What the server asks the user to confirm before a send (409 + action): "confirm_no_attachment" or "confirm_retry". */
export const actionOf = (err: unknown): string | null =>
  err instanceof ApiError && err.body && typeof err.body === 'object' && 'action' in err.body ? String((err.body as { action: unknown }).action) : null

export const suggestPersonalLine = (ctx: OutreachContext, variant: number) =>
  json<PersonalLineResult>('/outreach/personal-line', { ...post({ ...ctx, variant }), timeoutMs: SLOW_MS })
export const suggestAnswer = (ctx: OutreachContext, question: string, variant: number) =>
  json<AnswerResult>('/outreach/answer', { ...post({ ...ctx, question, variant }), timeoutMs: SLOW_MS })
export const regenerateEmailPart = (emailId: string, body: {
  part: 'subject' | 'body'; variant: number; personal_note?: string | null
  answers?: Array<{ question: string; answer: string }>; job_description?: string | null; job_title?: string | null
}) => json<{ part: string; subject?: string; body?: string; words?: number; notes?: string[] }>(`/emails/${emailId}/regenerate`, { ...post(body), timeoutMs: SLOW_MS })
export const emailAttachmentInfo = (emailId: string, template: string | null) =>
  json<AttachmentInfo>(`/emails/${emailId}/attachment-info${withTemplate(template)}`, { timeoutMs: 120_000 })
export const editEmail = (emailId: string, changes: { subject?: string; body?: string }) =>
  json<EmailRecord>(`/emails/${emailId}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(changes) })
export const sendEmail = (emailId: string, template: string | null, opts: { withoutAttachment?: boolean; confirmRetry?: boolean } = {}) => {
  const params = new URLSearchParams()
  if (template) params.set('template', template)
  if (opts.withoutAttachment) params.set('send_without_attachment', 'true')
  if (opts.confirmRetry) params.set('confirm_retry', 'true')
  const query = params.toString()
  return json<EmailRecord>(`/emails/${emailId}/approve${query ? `?${query}` : ''}`, { method: 'POST', timeoutMs: 120_000 })
}
export const rejectEmail = (emailId: string) => json<EmailRecord>(`/emails/${emailId}/reject`, { method: 'POST' })
