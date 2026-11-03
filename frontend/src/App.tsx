import {
  Briefcase,
  Check,
  FileText,
  Lock,
  Palette,
  RotateCcw,
  Send,
  Target,
  UserCheck,
  Wand2,
} from 'lucide-react'
import { RoleSyncIcon } from './components/RoleSyncLogo.tsx'
import { FlowProvider, useFlow } from './flow.tsx'
import { STEP_IDS, type StepId } from './flowModel.ts'
import { STEP_DETAILS, STEP_LABELS, lockReason } from './steps.ts'
import { DesignStep } from './steps/DesignStep.tsx'
import { DraftStep } from './steps/DraftStep.tsx'
import { EmailStep } from './steps/EmailStep.tsx'
import { FitStep } from './steps/FitStep.tsx'
import { JobStep } from './steps/JobStep.tsx'
import { UploadStep } from './steps/UploadStep.tsx'

const STEP_ICONS: Record<StepId, React.ComponentType<{ size?: number; className?: string }>> = {
  upload: FileText,
  job: Briefcase,
  fit: Target,
  draft: Wand2,
  design: Palette,
  email: Send,
}

function Shell() {
  const { flow, update, reset } = useFlow()
  const go = (step: StepId) => update({ step })
  const next = (from: StepId) => go(STEP_IDS[STEP_IDS.indexOf(from) + 1])
  // a reload can restore a step whose prerequisites are gone; fall back to the first step that is open
  const step = lockReason(flow.step, flow) ? 'upload' : flow.step
  const currentIndex = STEP_IDS.indexOf(step)
  const progressPercent = Math.round(((currentIndex + 1) / STEP_IDS.length) * 100)

  return (
    <div className="app min-h-screen">
      {/* Top Application Bar */}
      <header className="top border-b border-slate-800/80 pb-6 mb-8">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3.5">
            <RoleSyncIcon size={44} className="shrink-0 drop-shadow-[0_4px_16px_rgba(99,91,255,0.35)]" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">
                  Role<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Sync</span>
                </h1>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/15 text-cyan-300 border border-cyan-500/25">
                  AI Career Engine
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Align Your Profile. Accelerate Your Career.
              </p>
            </div>
          </div>

          {/* Active Context Indicators */}
          <div className="flex items-center gap-3">
            {flow.profile && (
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
                <UserCheck size={14} className="text-emerald-400" />
                <span className="font-medium text-white">{flow.profile.name}</span>
                {flow.job.company && (
                  <span className="text-slate-500">→ {flow.job.company}</span>
                )}
              </div>
            )}
            {flow.profile && (
              <button
                type="button"
                onClick={() => reset()}
                className="btn small ghost text-slate-400 hover:text-white"
                title="Start a new flow"
              >
                <RotateCcw size={13} className="mr-1 inline" />
                Reset
              </button>
            )}
          </div>
        </div>

        {/* Multi-step Visual Stepper */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span className="font-medium">
              Step {currentIndex + 1} of {STEP_IDS.length}:{' '}
              <span className="text-indigo-400 font-semibold">{STEP_DETAILS[step].subtitle}</span>
            </span>
            <span className="font-mono text-slate-500">{progressPercent}% complete</span>
          </div>

          {/* Progress Bar Track */}
          <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800/80">
            <div
              className="h-full bg-gradient-to-r from-indigo-600 via-indigo-500 to-sky-400 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <nav aria-label="Steps" className="pt-2">
            <ol className="steps flex gap-2 overflow-x-auto pb-1">
              {STEP_IDS.map((id, i) => {
                const reason = lockReason(id, flow)
                const isCurrent = id === step
                const isDone = i < currentIndex && !reason
                const IconComponent = STEP_ICONS[id]
                const details = STEP_DETAILS[id]

                return (
                  <li key={id} className="flex-1 min-w-[120px]">
                    <button
                      className={`step w-full text-left flex items-center gap-2.5 p-2.5 rounded-xl border transition-all ${
                        isCurrent
                          ? 'current bg-indigo-950/40 border-indigo-500/80 shadow-lg shadow-indigo-500/10 text-white'
                          : isDone
                          ? 'done bg-slate-900/40 border-emerald-500/30 text-slate-300 hover:border-slate-700'
                          : reason
                          ? 'opacity-40 bg-slate-950/40 border-slate-900 text-slate-500 cursor-not-allowed'
                          : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                      }`}
                      disabled={!!reason}
                      title={reason ?? details.desc}
                      aria-current={isCurrent ? 'step' : undefined}
                      onClick={() => go(id)}
                    >
                      <span
                        className={`num shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-xs font-semibold ${
                          isCurrent
                            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                            : isDone
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                            : 'bg-slate-800/80 text-slate-400'
                        }`}
                      >
                        {isDone ? <Check size={14} /> : reason ? <Lock size={12} /> : i + 1}
                      </span>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <IconComponent size={13} className={isCurrent ? 'text-indigo-400' : 'text-slate-400'} />
                          <span className="name text-xs font-semibold truncate">
                            {STEP_LABELS[id]}
                          </span>
                        </div>
                        <div className="hidden lg:block text-[10px] text-slate-400 truncate mt-0.5">
                          {details.subtitle}
                        </div>
                      </div>
                    </button>
                  </li>
                )
              })}
            </ol>
          </nav>
        </div>
      </header>

      {/* Main Step Container */}
      <main className="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 sm:p-8 backdrop-blur-xl shadow-2xl ring-1 ring-white/5">
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
  return (
    <FlowProvider>
      <Shell />
    </FlowProvider>
  )
}
