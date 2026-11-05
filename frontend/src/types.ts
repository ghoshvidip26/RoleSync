/** Shapes of the FastAPI responses this app reads (see app/models and app/main.py). */

export interface ProfileData {
  name: string
  email: string
  github: string | null
  portfolio: string | null
  linkedin: string | null
  summary: string | null
  skills: string[]
  programming_languages: string[]
  frameworks: string[]
  databases_tools: string[]
  experience: string[]
  projects: string[]
  achievements: string[]
  education: string[]
  target_roles: string[] | null
}

export interface Profile extends ProfileData {
  id: string
}

export interface AtsFlag {
  code: string
  severity: 'warning' | 'info'
  message: string
}

export interface OriginalAssessment {
  stored: boolean
  preserve_supported: boolean
  reason: string | null
  ats: { risk: 'low' | 'medium' | 'high'; flags: AtsFlag[] } | null
  contacts?: Record<string, string>
}

/** A job the user wants to apply to. Only `description` is needed today; a job link or a feed fills the rest later. */
export interface Job {
  title: string
  company: string
  link: string
  description: string
}

export interface ReviewResult {
  verdict: {
    fit_assessment: {
      verdict: string
      reasoning: string
      matching_requirements: string[]
      gaps: string[]
      uncertain_requirements: string[]
    }
  }
  tailoring: { needed: boolean; reasons: string[] }
}

export interface ExperienceItem {
  company: string
  job_title: string
  duration: string | null
  location: string | null
  bullets: string[]
}

export interface ProjectItem {
  name: string
  technologies: string[]
  description: string | null
  bullets: string[]
}

export interface ResumeDraft {
  headline: string | null
  summary: string | null
  skills: string[]
  experience: ExperienceItem[]
  projects: ProjectItem[]
  achievements: string[]
  education: Array<{ details: string }>
  recommendations: string[]
  claims_requiring_confirmation: string[]
}

export type DraftStatus = 'pending_review' | 'approved' | 'rejected'

export interface DraftRecord {
  id: string
  profile_id: string
  status: DraftStatus
  draft: ResumeDraft
}

export interface TemplateInfo {
  id: string
  name: string
  description: string
  best_for: string
  ats_notes: string
  sections: string[]
  preview_url: string
}

export interface OwnDesign {
  id: 'original'
  name: string
  available: boolean
  preserve_supported: boolean
  reason: string | null
  ats: OriginalAssessment['ats']
}

export interface Gallery {
  templates: TemplateInfo[]
  original: OwnDesign | null
}

export interface EmailRecord {
  id: string
  profile_id: string
  company_id: string
  recipient_id: string
  subject: string | null
  body: string | null
  status: string
  resume_draft_id: string | null
  error: string | null
}

export interface PersonalLineResult {
  text: string
  basis: 'website' | 'job_description' | 'none'
  source: string
  evidence: string[]
  limitation: string | null
  website_status: string
  website_reason: string | null
}

export interface AnswerResult {
  answer: string
  evidence: string[]
  source: string
  words: number
}

export interface AttachmentInfo {
  has_attachment: boolean
  filename: string | null
  size_bytes?: number
}

export interface PdfResult {
  url: string
  renderer: string
  validation: string
  pages: number
}
