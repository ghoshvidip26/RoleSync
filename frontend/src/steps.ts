import type { FlowState, StepId } from './flowModel.ts'

export const MIN_JOB_CHARS = 40

export const STEP_LABELS: Record<StepId, string> = {
  upload: 'Resume', job: 'Job', fit: 'Fit', draft: 'Tailor', design: 'Design', email: 'E-mail',
}

/** Why a step is locked (null = open). A step opens only when what it needs has been done. */
export function lockReason(step: StepId, flow: FlowState): string | null {
  const hasJob = flow.job.description.trim().length >= MIN_JOB_CHARS
  const approved = flow.draft?.status === 'approved'
  switch (step) {
    case 'upload': return null
    case 'job': return flow.profile ? null : 'Upload and confirm your resume first'
    case 'fit': case 'draft':
      return !flow.profile ? 'Upload and confirm your resume first' : !hasJob ? 'Paste the job description first' : null
    case 'design': return approved ? null : 'Approve your tailored resume first'
    case 'email': return !approved ? 'Approve your tailored resume first' : !flow.template ? 'Choose a design first' : null
  }
}
