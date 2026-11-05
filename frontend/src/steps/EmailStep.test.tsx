import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api.ts'
import { FlowProvider } from '../flow.tsx'
import { EMPTY_FLOW, FLOW_KEY } from '../flowModel.ts'
import type { EmailRecord, PersonalLineResult } from '../types.ts'
import { EmailStep } from './EmailStep.tsx'

vi.mock('../api.ts', async (original) => ({
  ...(await original<typeof import('../api.ts')>()),
  createCompany: vi.fn(), createRecipient: vi.fn(), createOutreach: vi.fn(), editEmail: vi.fn(), sendEmail: vi.fn(), rejectEmail: vi.fn(),
  emailAttachment: vi.fn(), emailAttachmentInfo: vi.fn(), regenerateEmailPart: vi.fn(), suggestPersonalLine: vi.fn(), suggestAnswer: vi.fn(),
}))

const PROFILE = { id: 'p1', name: 'Alex Doe', email: 'alex@example.com' }
const JD = 'Backend Engineer at Acme.\nWhy do you want to work at Acme?\nYou will build APIs.'
const EMAIL: EmailRecord = {
  id: 'e1', profile_id: 'p1', company_id: 'c1', recipient_id: 'r1', subject: 'Backend Engineer Opportunity | Alex',
  body: 'Hi Acme Team,\n\nFirst body.\n\nThanks & Regards,\nAlex', status: 'pending_review', resume_draft_id: 'd1', error: null,
}
const LINE: PersonalLineResult = {
  text: 'I read that Acme automates payments, and my work on StackSearch is relevant.', basis: 'job_description', source: 'template',
  evidence: ['Acme automates payments.'], limitation: null, website_status: 'none', website_reason: null,
}

const open = (over: Record<string, unknown> = {}) => {
  sessionStorage.setItem(FLOW_KEY, JSON.stringify({
    ...EMPTY_FLOW, step: 'email', profile: PROFILE, template: 'classic',
    draft: { id: 'd1', profile_id: 'p1', status: 'approved', draft: {} },
    job: { title: 'Backend Engineer', company: 'Acme', link: '', description: JD }, ...over,
  }))
  render(<FlowProvider><EmailStep onRestart={vi.fn()} /></FlowProvider>)
}
const box = (name: string) => screen.getByLabelText(name) as HTMLInputElement | HTMLTextAreaElement
const button = (name: RegExp | string) => screen.getByRole('button', { name })

async function fillRecipient(email = 'team@acme.com') {
  await userEvent.type(box("Recipient's e-mail"), email)
}
async function writeEmail() {
  await fillRecipient()
  await userEvent.click(button(/Write my e-mail/))
  await screen.findByText('Review your e-mail')
}

beforeEach(() => {
  sessionStorage.clear()
  for (const fn of Object.values(api)) if (vi.isMockFunction(fn)) fn.mockReset()
  vi.mocked(api.createCompany).mockResolvedValue({ id: 'c1' })
  vi.mocked(api.createRecipient).mockResolvedValue({ id: 'r1' })
  vi.mocked(api.createOutreach).mockResolvedValue(EMAIL)
  vi.mocked(api.emailAttachmentInfo).mockResolvedValue({ has_attachment: true, filename: 'Alex_Doe_Resume.pdf', size_bytes: 9000 })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

describe('before the e-mail is written', () => {
  it('prefills the company from the job and needs a valid recipient address', async () => {
    open()
    expect(box('Company').value).toBe('Acme')
    expect(button(/Write my e-mail/)).toBeDisabled()
    await fillRecipient('nope')
    expect(button(/Write my e-mail/)).toBeDisabled()
    await userEvent.type(box("Recipient's e-mail"), '@acme.com')
    expect(button(/Write my e-mail/)).toBeEnabled()
  })

  it('works without a recipient name or website', async () => {
    open()
    await writeEmail()
    expect(api.createCompany).toHaveBeenCalledWith('Acme', null)
    expect(api.createRecipient).toHaveBeenCalledWith(null, 'team@acme.com', null)
  })

  it('suggests a personal line, shows what it is based on, and lets the user edit it', async () => {
    vi.mocked(api.suggestPersonalLine).mockResolvedValue(LINE)
    open()
    await fillRecipient()
    await userEvent.click(button('Suggest a personal line'))
    await waitFor(() => expect(box('One personal line (optional)').value).toBe(LINE.text))
    expect(screen.getByText('AI suggestion')).toBeInTheDocument()
    expect(screen.getByText(/Based on the job description/)).toBeInTheDocument()
    await userEvent.type(box('One personal line (optional)'), ' Edited.')
    expect(screen.getByText('Edited by you')).toBeInTheDocument()
  })

  it('does not replace an edited personal line when the user declines, and does when they agree', async () => {
    vi.mocked(api.suggestPersonalLine).mockResolvedValueOnce(LINE).mockResolvedValueOnce({ ...LINE, text: 'A newer line about Acme and StackSearch work.' })
    open()
    await fillRecipient()
    await userEvent.click(button('Suggest a personal line'))
    await waitFor(() => expect(box('One personal line (optional)').value).toBe(LINE.text))
    await userEvent.type(box('One personal line (optional)'), ' mine')
    vi.mocked(window.confirm).mockReturnValueOnce(false)
    await userEvent.click(button('Regenerate personal line'))
    expect(api.suggestPersonalLine).toHaveBeenCalledOnce()
    expect(box('One personal line (optional)').value).toContain('mine')
    await userEvent.click(button('Regenerate personal line'))
    await waitFor(() => expect(box('One personal line (optional)').value).toBe('A newer line about Acme and StackSearch work.'))
    expect(vi.mocked(api.suggestPersonalLine).mock.calls[1][1]).toBe(1)                       // the next variant
  })

  it('tells the user when the website could not be read', async () => {
    vi.mocked(api.suggestPersonalLine).mockResolvedValue({ ...LINE, limitation: 'The company website could not be read (timed out).' })
    open()
    await fillRecipient()
    await userEvent.click(button('Suggest a personal line'))
    expect(await screen.findByText(/website could not be read/)).toBeInTheDocument()
  })

  it('offers questions found in the job description and writes an answer with its evidence', async () => {
    vi.mocked(api.suggestAnswer).mockResolvedValue({ answer: 'I build APIs.', evidence: ['At PayLoop: cut latency 45%'], source: 'model', words: 3 })
    open()
    await fillRecipient()
    await userEvent.click(button('+ Why do you want to work at Acme?'))
    expect(await screen.findByDisplayValue('I build APIs.')).toBeInTheDocument()
    expect(screen.getByText('At PayLoop: cut latency 45%')).toBeInTheDocument()
    expect(screen.getByText('3/45 words')).toBeInTheDocument()
  })

  it('keeps answers out of the e-mail unless the user adds them', async () => {
    vi.mocked(api.suggestAnswer).mockResolvedValue({ answer: 'I build APIs.', evidence: [], source: 'model', words: 3 })
    open()
    await fillRecipient()
    await userEvent.click(button('+ Why should you choose me?'))
    await screen.findByDisplayValue('I build APIs.')
    await userEvent.click(button(/Write my e-mail/))
    await screen.findByText('Review your e-mail')
    expect(vi.mocked(api.createOutreach).mock.calls[0][0].answers).toEqual([])
  })

  it('sends only the answers the user added, exactly as edited', async () => {
    vi.mocked(api.suggestAnswer).mockResolvedValue({ answer: 'I build APIs.', evidence: [], source: 'model', words: 3 })
    open()
    await fillRecipient()
    await userEvent.click(button('+ Why should you choose me?'))
    const answer = await screen.findByDisplayValue('I build APIs.')
    await userEvent.type(answer, ' Really.')
    await userEvent.click(screen.getByRole('checkbox'))
    await userEvent.click(button(/Write my e-mail/))
    await screen.findByText('Review your e-mail')
    expect(vi.mocked(api.createOutreach).mock.calls[0][0].answers).toEqual([{ question: 'Why should you choose me?', answer: 'I build APIs. Really.' }])
  })

  it('saves the company and recipient once, however many things are generated', async () => {
    vi.mocked(api.suggestPersonalLine).mockResolvedValue(LINE)
    open()
    await fillRecipient()
    await userEvent.click(button('Suggest a personal line'))
    await waitFor(() => expect(box('One personal line (optional)').value).toBe(LINE.text))
    await userEvent.click(button(/Write my e-mail/))
    await screen.findByText('Review your e-mail')
    expect(api.createCompany).toHaveBeenCalledTimes(1)
    expect(api.createRecipient).toHaveBeenCalledTimes(1)
  })

  it('warns when there is no approved resume, and still lets the user write', async () => {
    open({ draft: null, template: null })
    expect(screen.getByText(/not approved, so this e-mail would go out without a resume/)).toBeInTheDocument()
    await fillRecipient()
    await userEvent.click(button(/Write my e-mail/))
    await screen.findByText('Review your e-mail')
    expect(vi.mocked(api.createOutreach).mock.calls[0][0].resume_draft_id).toBeNull()
  })
})

describe('reviewing and regenerating', () => {
  it('regenerating the subject leaves the body, and the user’s edits to it, alone', async () => {
    vi.mocked(api.regenerateEmailPart).mockResolvedValue({ part: 'subject', subject: 'Exploring Backend Opportunities at Acme' })
    open()
    await writeEmail()
    await userEvent.type(box('E-mail'), ' MY EDIT')
    await userEvent.click(button('Regenerate subject'))
    await waitFor(() => expect(box('Subject').value).toBe('Exploring Backend Opportunities at Acme'))
    expect(box('E-mail').value).toContain('MY EDIT')
    expect(api.regenerateEmailPart).toHaveBeenCalledWith('e1', expect.objectContaining({ part: 'subject', variant: 1 }))
  })

  it('regenerating the body leaves the subject alone and passes the personal line and added answers', async () => {
    vi.mocked(api.regenerateEmailPart).mockResolvedValue({ part: 'body', body: 'Hi Acme Team,\n\nNew body.\n\nThanks & Regards,\nAlex' })
    open()
    await writeEmail()
    await userEvent.clear(box('Subject'))
    await userEvent.type(box('Subject'), 'My own subject')
    await userEvent.click(button('Regenerate body'))
    await waitFor(() => expect(box('E-mail').value).toContain('New body.'))
    expect(box('Subject').value).toBe('My own subject')
    expect(api.regenerateEmailPart).toHaveBeenCalledWith('e1', expect.objectContaining({ part: 'body', answers: [] }))
  })

  it('asks before replacing a manually edited part and keeps it when declined', async () => {
    open()
    await writeEmail()
    await userEvent.type(box('E-mail'), ' MINE')
    vi.mocked(window.confirm).mockReturnValueOnce(false)
    await userEvent.click(button('Regenerate body'))
    expect(api.regenerateEmailPart).not.toHaveBeenCalled()
    expect(box('E-mail').value).toContain('MINE')
  })

  it('shows a clear error when regeneration fails and keeps the text', async () => {
    vi.mocked(api.regenerateEmailPart).mockRejectedValue(new Error('Could not write a new body: model down'))
    open()
    await writeEmail()
    await userEvent.click(button('Regenerate body'))
    expect(await screen.findByText(/model down/)).toBeInTheDocument()
    expect(box('E-mail').value).toContain('First body.')
  })

  it('shows the word count of the body before the signature', async () => {
    open()
    await writeEmail()
    expect(screen.getByText(/5 words before the signature/)).toBeInTheDocument()
  })
})

describe('sending', () => {
  it('shows the exact attachment file name before the final confirmation', async () => {
    vi.mocked(api.sendEmail).mockResolvedValue({ ...EMAIL, status: 'sent' })
    open()
    await writeEmail()
    expect(await screen.findByText('Alex_Doe_Resume.pdf')).toBeInTheDocument()
    await userEvent.click(button('Send now'))
    await waitFor(() => expect(api.sendEmail).toHaveBeenCalled())
    expect(vi.mocked(window.confirm).mock.calls.at(-1)![0]).toContain('“Alex_Doe_Resume.pdf”')
    expect(api.sendEmail).toHaveBeenCalledWith('e1', 'classic', { withoutAttachment: false })
    expect(await screen.findByText('Outreach Delivered!')).toBeInTheDocument()
  })

  it('sends nothing when the user cancels the final confirmation', async () => {
    open()
    await writeEmail()
    vi.mocked(window.confirm).mockReturnValue(false)
    await userEvent.click(button('Send now'))
    expect(api.sendEmail).not.toHaveBeenCalled()
  })

  it('needs two confirmations to send without an approved resume, and says so on the server call', async () => {
    vi.mocked(api.createOutreach).mockResolvedValue({ ...EMAIL, resume_draft_id: null })
    vi.mocked(api.sendEmail).mockResolvedValue({ ...EMAIL, resume_draft_id: null, status: 'sent' })
    open({ draft: null, template: null })
    await writeEmail()
    expect(screen.getByText(/No resume will be attached/)).toBeInTheDocument()
    vi.mocked(window.confirm).mockClear()
    await userEvent.click(button('Send now'))
    await waitFor(() => expect(api.sendEmail).toHaveBeenCalled())
    expect(window.confirm).toHaveBeenCalledTimes(2)
    expect(api.sendEmail).toHaveBeenCalledWith('e1', null, { withoutAttachment: true })
  })

  it('does not send without an attachment when the user declines the second confirmation', async () => {
    vi.mocked(api.createOutreach).mockResolvedValue({ ...EMAIL, resume_draft_id: null })
    open({ draft: null, template: null })
    await writeEmail()
    vi.mocked(window.confirm).mockReturnValueOnce(true).mockReturnValueOnce(false)
    await userEvent.click(button('Send now'))
    expect(api.sendEmail).not.toHaveBeenCalled()
  })

  it('keeps the e-mail after a failed send and sends again only after the user confirms', async () => {
    vi.mocked(api.sendEmail)
      .mockResolvedValueOnce({ ...EMAIL, status: 'failed', error: 'SMTP down' })
      .mockRejectedValueOnce(Object.assign(new api.ApiError('The previous attempt failed (SMTP down). Check it, then confirm.', 409, { action: 'confirm_retry' })))
      .mockResolvedValueOnce({ ...EMAIL, status: 'sent' })
    open()
    await writeEmail()
    await userEvent.click(button('Send now'))
    expect((await screen.findAllByText(/SMTP down/)).length).toBeGreaterThan(0)
    expect(box('E-mail').value).toContain('First body.')
    await userEvent.click(button('Try sending again'))
    await waitFor(() => expect(api.sendEmail).toHaveBeenCalledTimes(3))
    expect(vi.mocked(api.sendEmail).mock.calls[2][2]).toEqual(expect.objectContaining({ confirmRetry: true }))
    expect(await screen.findByText('Outreach Delivered!')).toBeInTheDocument()
  })

  it('disables Send while sending so it cannot be clicked twice', async () => {
    let finish: (e: EmailRecord) => void = () => {}
    vi.mocked(api.sendEmail).mockReturnValue(new Promise<EmailRecord>((r) => { finish = r }))
    open()
    await writeEmail()
    await userEvent.click(button('Send now'))
    await waitFor(() => expect(button(/Sending/)).toBeDisabled())
    finish({ ...EMAIL, status: 'sent' })
    await screen.findByText('Outreach Delivered!')
    expect(api.sendEmail).toHaveBeenCalledTimes(1)
  })
})
