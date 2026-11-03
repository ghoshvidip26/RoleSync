import { FlowProvider, useFlow } from './flow.tsx'
import { STEP_IDS, type StepId } from './flowModel.ts'
import { lockReason, STEP_LABELS } from './steps.ts'
import { DesignStep } from './steps/DesignStep.tsx'
import { DraftStep } from './steps/DraftStep.tsx'
import { EmailStep } from './steps/EmailStep.tsx'
import { FitStep } from './steps/FitStep.tsx'
import { JobStep } from './steps/JobStep.tsx'
import { UploadStep } from './steps/UploadStep.tsx'

function Shell() {
  const { flow, update, reset } = useFlow()
  const go = (step: StepId) => update({ step })
  const next = (from: StepId) => go(STEP_IDS[STEP_IDS.indexOf(from) + 1])
  // a reload can restore a step whose prerequisites are gone; fall back to the first step that is open
  const step = lockReason(flow.step, flow) ? 'upload' : flow.step

  return (
    <div className="app">
      <header className="top">
        <h1>Founders Outreach</h1>
        <nav aria-label="Steps">
          <ol className="steps">
            {STEP_IDS.map((id, i) => {
              const reason = lockReason(id, flow)
              return (
                <li key={id}>
                  <button className={`step ${id === step ? 'current' : ''}`} disabled={!!reason} title={reason ?? undefined}
                    aria-current={id === step ? 'step' : undefined} onClick={() => go(id)}>
                    <span className="num">{i + 1}</span><span className="name">{STEP_LABELS[id]}</span>
                  </button>
                </li>
              )
            })}
          </ol>
        </nav>
      </header>
      <main>
        {step === 'upload' && <UploadStep onNext={() => next('upload')} />}
        {step === 'job' && <JobStep onNext={() => next('job')} onSkipFit={() => go('draft')} />}
        {step === 'fit' && <FitStep onNext={() => next('fit')} />}
        {step === 'draft' && <DraftStep onNext={() => next('draft')} />}
        {step === 'design' && <DesignStep onNext={() => next('design')} />}
        {step === 'email' && <EmailStep onRestart={() => reset()} />}
      </main>
    </div>
  )
}

export default function App() {
  return <FlowProvider><Shell /></FlowProvider>
}
