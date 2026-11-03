import { useEffect, useRef, useState } from 'react'
import {
  ArrowRight,
  Eye,
  FileCheck,
  Layers,
  Sparkles,
} from 'lucide-react'
import { ChooseTemplateError, getGallery, resumePdf, templatePreviewUrl } from '../api.ts'
import { useFlow } from '../flow.tsx'
import { errorText } from '../lib.ts'
import type { Gallery, PdfResult } from '../types.ts'
import { Badge, ErrorBanner, Notice, StepHeader, Working } from '../ui.tsx'

export function DesignStep({ onNext }: { onNext: () => void }) {
  const { flow, update } = useFlow()
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [selected, setSelected] = useState<string | null>(flow.template)
  const [pdf, setPdf] = useState<(PdfResult & { template: string }) | null>(null)
  const [busy, setBusy] = useState<'gallery' | 'pdf' | null>(
    flow.profile && flow.draft?.status === 'approved' ? 'gallery' : null
  )
  const [error, setError] = useState<string | null>(null)
  const [failed, setFailed] = useState<Array<{ name: string; detail: string }>>([])
  const current = useRef<string | null>(null)
  const profileId = flow.profile?.id
  const draft = flow.draft

  useEffect(() => {
    if (!profileId || draft?.status !== 'approved') return // nothing to show until the draft is approved
    let live = true
    getGallery(profileId)
      .then((g) => {
        if (!live) return
        setGallery(g)
        setSelected((now) => now ?? (g.original?.available && g.original.preserve_supported ? 'original' : null))
      })
      .catch((err) => {
        if (live) setError(errorText(err))
      })
      .finally(() => {
        if (live) setBusy(null)
      })
    return () => {
      live = false
    }
  }, [profileId, draft?.status])
  useEffect(() => () => {
    if (current.current) URL.revokeObjectURL(current.current)
  }, [])

  const preview = async (template: string) => {
    if (!draft) return
    setSelected(template)
    setBusy('pdf')
    setError(null)
    setFailed([])
    try {
      const result = await resumePdf(draft.id, template)
      if (current.current) URL.revokeObjectURL(current.current)
      current.current = result.url
      setPdf({ ...result, template })
      update({ template })
    } catch (err) {
      setPdf(null)
      update({ template: null })
      if (err instanceof ChooseTemplateError) setFailed(err.failedChecks)
      setError(errorText(err))
    } finally {
      setBusy(null)
    }
  }

  // Show a design straight away: the one chosen before, else the user's own resume design when it can be reproduced.
  // That is the "keep my format" default: nothing to find or click before continuing.
  const opened = useRef(false)
  useEffect(() => {
    if (!gallery || opened.current || draft?.status !== 'approved') return
    opened.current = true
    const own = gallery.original
    const first = flow.template ?? (own?.available && own.preserve_supported ? 'original' : null)
    if (first) void Promise.resolve().then(() => preview(first))
  }, [gallery]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!draft || draft.status !== 'approved') {
    return (
      <section className="space-y-4">
        <StepHeader title="Choose a design" />
        <Notice tone="warn">Approve your tailored resume first.</Notice>
      </section>
    )
  }

  const own = gallery?.original
  return (
    <section className="space-y-6">
      <StepHeader
        title="Choose a design"
        intro="Your resume keeps its content; only the look changes. Every option is built to be read correctly by applicant tracking systems."
      />

      {busy === 'gallery' && <Working label="Loading designs" />}
      <ErrorBanner message={error} />
      {failed.length > 0 && (
        <Notice tone="warn">
          {failed.map((f) => (
            <div key={f.name}>
              {f.name}: {f.detail}
            </div>
          ))}
        </Notice>
      )}
      {busy === 'pdf' && (
        <Working label="Building your PDF" hint="Checking that nothing was lost or overlaps." />
      )}

      {pdf && busy !== 'pdf' && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h3 className="m-0 text-base font-bold text-white flex items-center gap-2">
                <FileCheck size={18} className="text-emerald-400" />
                {pdf.template === 'original' ? 'Your resume, in your own design' : 'Your resume'}
              </h3>
              <p className="small muted mt-1">
                {pdf.pages} page{pdf.pages === 1 ? '' : 's'} · {pdf.renderer.replace('template:', 'template ')} ·{' '}
                {pdf.validation === 'pass' ? (
                  <Badge tone="good">all checks passed</Badge>
                ) : (
                  <Badge tone="warn">passed with notes</Badge>
                )}
              </p>
            </div>
            <button className="btn primary" onClick={onNext}>
              {pdf.template === 'original' ? "Continue with my resume's design" : 'Use this design'}
              <ArrowRight size={15} />
            </button>
          </div>
          <iframe className="pdf rounded-xl shadow-xl" title="Tailored resume preview" src={pdf.url} />
        </div>
      )}

      {gallery && (
        <div className="space-y-4 pt-2">
          <h3 className="text-base font-semibold text-white flex items-center gap-2 m-0">
            <Layers size={17} className="text-indigo-400" />
            {pdf ? 'Or choose another design' : 'Designs'}
          </h3>

          <div className="cards">
            {own && (
              <article
                className={`card pick ${selected === 'original' ? 'selected' : ''} ${
                  own.available && own.preserve_supported ? '' : 'disabled'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <h4 className="m-0 text-white font-semibold">Your resume's own design</h4>
                  {own.available && own.preserve_supported && <Badge tone="good">default</Badge>}
                </div>
                <p className="small muted">
                  {own.available && own.preserve_supported
                    ? 'The look of the resume you uploaded: same layout, fonts and spacing, with your approved text.'
                    : `Not available: ${own.reason ?? 'no resume PDF was stored'}. Pick one of the templates instead.`}
                </p>
                <button
                  className={`btn ${own.available && own.preserve_supported ? 'primary' : ''}`}
                  disabled={busy !== null || !own.available || !own.preserve_supported}
                  onClick={() => preview('original')}
                >
                  <Sparkles size={14} />
                  Keep my design
                </button>
              </article>
            )}

            {gallery.templates.map((t) => (
              <article key={t.id} className={`card pick ${selected === t.id ? 'selected' : ''}`}>
                <img
                  className="thumb"
                  src={templatePreviewUrl(t.id)}
                  alt={`${t.name} template preview`}
                  loading="lazy"
                />
                <h4 className="m-0 text-white font-semibold">{t.name}</h4>
                <p className="small muted">{t.best_for}</p>
                <button className="btn" disabled={busy !== null} onClick={() => preview(t.id)}>
                  <Eye size={14} />
                  Preview with my resume
                </button>
              </article>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
