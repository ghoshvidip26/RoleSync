import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { EMPTY_FLOW, FLOW_KEY, type FlowState } from './flowModel.ts'
function load(): FlowState {
  try {
    const raw = sessionStorage.getItem(FLOW_KEY)
    return raw ? { ...EMPTY_FLOW, ...(JSON.parse(raw) as Partial<FlowState>) } : EMPTY_FLOW
  } catch {
    return EMPTY_FLOW
  }
}

interface Ctx {
  flow: FlowState
  update: (changes: Partial<FlowState>) => void
  reset: () => void
}

const FlowContext = createContext<Ctx | null>(null)

export function FlowProvider({ children }: { children: ReactNode }) {
  const [flow, setFlow] = useState<FlowState>(load)
  useEffect(() => {
    try {
      sessionStorage.setItem(FLOW_KEY, JSON.stringify(flow))
    } catch {
      /* storage may be unavailable (private window); the flow still works for this page view */
    }
  }, [flow])
  const value = useMemo<Ctx>(
    () => ({ flow, update: (changes) => setFlow((f) => ({ ...f, ...changes })), reset: () => setFlow(EMPTY_FLOW) }),
    [flow],
  )
  return <FlowContext.Provider value={value}>{children}</FlowContext.Provider>
}

export function useFlow(): Ctx {
  const ctx = useContext(FlowContext)
  if (!ctx) throw new Error('useFlow must be used inside FlowProvider')
  return ctx
}
