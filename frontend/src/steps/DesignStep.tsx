import { useEffect, useRef, useState } from 'react'
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
  const [busy, setBusy] = useState<'gallery' | 'pdf' | null>(flow.profile ? 'gallery' : null)
  const [error, setError] = useState<string | null>(null)
  const [failed, setFailed] = useState<Array<{ name: string; detail: string }>>([])
  const current = useRef<string | null>(null)
  const profileId = flow.profile?.id
  const draft = flow.draft

  useEffect(() => {
    if (!profileId) return
    let live = true
    getGallery(profileId)
      .then((g) => {
        if (!live) return
        setGallery(g)
        setSelected((now) => now ?? (g.original?.available && g.original.preserve_supported ? 'original' : null))
      })
      .catch((err) => { if (live) setError(errorText(err)) })
      .finally(() => { if (live) setBusy(null) })
    return () => { live = false }
  }, [profileId])
  useEffect(() => () => { if (current.current) URL.revokeObjectURL(current.current) }, [])

  const preview = async (template: string) => {
    if (!draft) return
    setSelected(template); setBusy('pdf'); setError(null); setFailed([])
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
    } finally { setBusy(null) }
  }

  if (!draft || draft.status !== 'approved') {
    return <section><StepHeader title="Choose a design" /><Notice tone="warn">Approve your tailored resume first.</Notice></section>
  }

  const own = gallery?.original
  return (
    <section>
      <StepHeader title="Choose a design" intro="Your resume keeps its content; only the look changes. Every option is built to be read correctly by applicant tracking systems." />
      {busy === 'gallery' && <Working label="Loading designs" />}
      <ErrorBanner message={error} />
      {failed.length > 0 && <Notice tone="warn">{failed.map((f) => <div key={f.name}>{f.name}: {f.detail}</div>)}</Notice>}
      {gallery && (
        <div className="cards">
          {own && (
            <article className={`card pick ${selected === 'original' ? 'selected' : ''} ${own.available && own.preserve_supported ? '' : 'disabled'}`}>
              <h4>Your own design {own.available && own.preserve_supported && <Badge tone="good">available</Badge>}</h4>
              <p className="small muted">{own.available && own.preserve_supported
                ? 'Rebuilt from your uploaded resume so it looks the same.'
                : `Not available: ${own.reason ?? 'no resume PDF was stored'}.`}</p>
              <button className="btn" disabled={busy !== null || !own.available || !own.preserve_supported} onClick={() => preview('original')}>Preview</button>
            </article>
          )}
          {gallery.templates.map((t) => (
            <article key={t.id} className={`card pick ${selected === t.id ? 'selected' : ''}`}>
              <img className="thumb" src={templatePreviewUrl(t.id)} alt={`${t.name} template preview`} loading="lazy" />
              <h4>{t.name}</h4>
              <p className="small muted">{t.best_for}</p>
              <button className="btn" disabled={busy !== null} onClick={() => preview(t.id)}>Preview with my resume</button>
            </article>
          ))}
        </div>
      )}
      {busy === 'pdf' && <Working label="Building your PDF" hint="Checking that nothing was lost or overlaps." />}
      {pdf && busy !== 'pdf' && (
        <div className="stack">
          <h3>Your resume</h3>
          <p className="small muted">
            {pdf.pages} page{pdf.pages === 1 ? '' : 's'} · {pdf.renderer.replace('template:', 'template ')} ·{' '}
            {pdf.validation === 'pass' ? <Badge tone="good">all checks passed</Badge> : <Badge tone="warn">passed with notes</Badge>}
          </p>
          <iframe className="pdf" title="Tailored resume preview" src={pdf.url} />
          <div className="actions"><button className="btn primary" onClick={onNext}>Use this design</button></div>
        </div>
      )}
    </section>
  )
}
