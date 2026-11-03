import type { SVGProps } from 'react'

export function RoleSyncIcon({
  size = 36,
  className = '',
  ...props
}: SVGProps<SVGSVGElement> & { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 44 44"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      {...props}
    >
      <defs>
        {/* Primary Sync Gradient: Electric Indigo to Neon Cyan */}
        <linearGradient id="rs-sync-grad" x1="4" y1="4" x2="40" y2="40" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#635BFF" />
          <stop offset="50%" stopColor="#818CF8" />
          <stop offset="100%" stopColor="#22D3EE" />
        </linearGradient>

        {/* Glow Filter */}
        <filter id="rs-glow" x="0" y="0" width="44" height="44" filterUnits="userSpaceOnUse">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>

        {/* Sparkle Gradient */}
        <linearGradient id="rs-spark-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="100%" stopColor="#22D3EE" />
        </linearGradient>
      </defs>

      {/* Background Badge with subtle border */}
      <rect
        width="44"
        height="44"
        rx="12"
        fill="#11182B"
        stroke="#25304A"
        strokeWidth="1.5"
      />

      {/* S-Sync Flow Lines (Double Loop Sync Mark) */}
      <g filter="url(#rs-glow)">
        {/* Upper S-Loop Arc */}
        <path
          d="M 14 19 C 14 13.5, 18.5 10.5, 24 10.5 C 27 10.5, 29.5 11.5, 31 13"
          stroke="url(#rs-sync-grad)"
          strokeWidth="3.2"
          strokeLinecap="round"
        />

        {/* Diagonal S Cross Bridge */}
        <path
          d="M 28 15 C 27 18, 17 22, 16 28"
          stroke="url(#rs-sync-grad)"
          strokeWidth="3.2"
          strokeLinecap="round"
        />

        {/* Lower S-Loop Arc */}
        <path
          d="M 13 30 C 14.5 32, 17.5 33.5, 21 33.5 C 26.5 33.5, 30.5 30, 30.5 24.5"
          stroke="url(#rs-sync-grad)"
          strokeWidth="3.2"
          strokeLinecap="round"
        />

        {/* Sync Arrowheads to emphasize active alignment cycle */}
        <path
          d="M 28.5 10 L 32 13.5 L 28.5 17"
          stroke="#22D3EE"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M 15.5 27 L 12 30.5 L 15.5 34"
          stroke="#635BFF"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>

      {/* Distinctive AI 4-Point Sparkle at Top-Right */}
      <path
        d="M 32 4.5 Q 32 8.5 36 8.5 Q 32 8.5 32 12.5 Q 32 8.5 28 8.5 Q 32 8.5 32 4.5 Z"
        fill="url(#rs-spark-grad)"
      />
      <circle cx="32" cy="8.5" r="0.8" fill="#FFFFFF" />
    </svg>
  )
}

export function RoleSyncLogo({
  size = 40,
  showTagline = true,
  className = '',
}: {
  size?: number
  showTagline?: boolean
  className?: string
}) {
  return (
    <div className={`flex items-center gap-3.5 ${className}`}>
      <RoleSyncIcon size={size} className="shrink-0 drop-shadow-[0_4px_16px_rgba(99,91,255,0.25)]" />
      <div>
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold tracking-tight text-white font-['Plus_Jakarta_Sans',sans-serif]">
            Role<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Sync</span>
          </span>
          <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-cyan-300 border border-cyan-500/20">
            AI Career Engine
          </span>
        </div>
        {showTagline && (
          <p className="text-xs text-slate-400 font-medium tracking-normal mt-0.5">
            Align Your Profile. Accelerate Your Career.
          </p>
        )}
      </div>
    </div>
  )
}

