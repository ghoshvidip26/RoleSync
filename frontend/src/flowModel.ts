import type { DraftRecord, Job, OriginalAssessment, Profile, ReviewResult, RoleRecommendation } from './types.ts'

/** Everything the user has done so far. Files cannot be saved, so a reload keeps the ids and text only. */
export interface FitResult {
  jobDescription: string // the text these results belong to; a different description makes them stale
  roles: RoleRecommendation[] | null
  review: ReviewResult | null
}

/** What the AI last suggested for the positioning fields of one draft, so edits can be told apart from suggestions
 * (and survive leaving the page). `variants` counts how many times each field was regenerated. */
export interface PositioningBaseline {
  draftId: string
  headline: string
  summary: string
  skills: string[]
  variants: { headline: number; summary: number; skills: number }
}

export interface FlowState {
  step: StepId
  profile: Profile | null
  assessment: OriginalAssessment | null
  job: Job
  fit: FitResult | null
  draft: DraftRecord | null
  positioning: PositioningBaseline | null
  template: string | null // a gallery id, "original", or null (not chosen yet)
  emailId: string | null
}

export const EMPTY_JOB: Job = { title: '', company: '', link: '', description: '' }
export const STEP_IDS = ['upload', 'job', 'fit', 'draft', 'design', 'email'] as const
export type StepId = (typeof STEP_IDS)[number]

export const EMPTY_FLOW: FlowState = {
  step: 'upload', profile: null, assessment: null, job: EMPTY_JOB, fit: null, draft: null, positioning: null, template: null, emailId: null,
}

export const FLOW_KEY = 'founders-outreach-flow-v1'
