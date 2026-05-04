import { useEffect, useRef, useState } from 'react';
import SlashMark from './SlashMark';

const INK = '#0e0e0e';
const INK_2 = '#2c2c2c';
const MUTE = '#707070';
const LINE = '#d8d6cf';
const ACCENT = '#d52b1e';
const BG = '#fafaf7';
const PAPER_BG_INV = '#f5f3ec';

function Secbar({ num, ttl, meta, id }: { num: string; ttl: string; meta?: string; id?: string }) {
  return (
    <div className="secbar" id={id}>
      <span className="num">§ {num}</span>
      <span className="ttl">// {ttl}</span>
      <span className="meta">{meta ?? ''}</span>
    </div>
  );
}

function Callout({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="callout" style={{ gridColumn: '1 / -1' }}>
      <div className="lbl">// {label}</div>
      <p>{children}</p>
    </div>
  );
}

function Code({ children }: { children: string }) {
  return <code className="code-inline">{children}</code>;
}

// --- VISUALIZATION 1: Staircase ---
function StaircaseViz() {
  const [isTransformed, setIsTransformed] = useState(false);

  return (
    <div className="viz" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div
        style={{
          display: 'flex',
          gap: '48px',
          alignItems: 'center',
          flexWrap: 'wrap',
          justifyContent: 'center',
          width: '100%',
        }}
      >
        <svg
          width="260"
          height="260"
          viewBox="0 0 300 300"
          style={{
            overflow: 'visible',
            transition: 'transform 1s ease',
            transform: isTransformed ? 'rotate(-45deg)' : 'rotate(0deg)',
          }}
        >
          <defs>
            <pattern id="stair-grid" width="30" height="30" patternUnits="userSpaceOnUse">
              <path d="M 30 0 L 0 0 0 30" fill="none" stroke={LINE} strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="300" height="300" fill="url(#stair-grid)" />

          {/* Shaded regions */}
          <polygon points="0,300 0,0 300,0" fill={INK} fillOpacity="0.05" />
          <polygon points="0,300 300,0 300,300" fill={ACCENT} fillOpacity="0.06" />

          {/* True diagonal boundary */}
          <line x1="0" y1="300" x2="300" y2="0" stroke={INK} strokeWidth="2" strokeDasharray="6 4" />

          {/* Staircase approximation */}
          <path
            d="M 0 300 L 0 270 L 30 270 L 30 240 L 60 240 L 60 210 L 90 210 L 90 180 L 120 180 L 120 150 L 150 150 L 150 120 L 180 120 L 180 90 L 210 90 L 210 60 L 240 60 L 240 30 L 270 30 L 270 0 L 300 0"
            fill="none"
            stroke={ACCENT}
            strokeWidth="3"
            style={{ opacity: isTransformed ? 0 : 1, transition: 'opacity 0.5s ease' }}
          />

          {/* Single cut */}
          <line
            x1="-100"
            y1="150"
            x2="400"
            y2="150"
            stroke={ACCENT}
            strokeWidth="3"
            style={{ opacity: isTransformed ? 1 : 0, transition: 'opacity 0.5s ease' }}
          />

          <text
            x="80"
            y="100"
            fontSize="11"
            fontFamily="JetBrains Mono, monospace"
            fill={INK}
            letterSpacing="2"
            style={{ opacity: isTransformed ? 0 : 0.6, transition: 'opacity 0.5s' }}
          >
            CLASS A
          </text>
          <text
            x="180"
            y="230"
            fontSize="11"
            fontFamily="JetBrains Mono, monospace"
            fill={ACCENT}
            letterSpacing="2"
            style={{ opacity: isTransformed ? 0 : 0.6, transition: 'opacity 0.5s' }}
          >
            CLASS B
          </text>
        </svg>

        <div style={{ maxWidth: '240px' }}>
          <div
            style={{
              fontFamily: '"Helvetica Neue", "Inter Tight", Arial, sans-serif',
              fontSize: '60px',
              fontWeight: 700,
              letterSpacing: '-0.025em',
              lineHeight: 1,
              color: isTransformed ? ACCENT : INK,
              transition: 'color 0.3s',
            }}
          >
            {isTransformed ? '1' : '19'}
          </div>
          <div
            style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px',
              color: MUTE,
              textTransform: 'uppercase',
              letterSpacing: '0.14em',
              marginTop: '4px',
              marginBottom: '14px',
            }}
          >
            // splits needed
          </div>
          <p
            style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px',
              color: INK_2,
              lineHeight: 1.7,
              margin: 0,
            }}
          >
            {isTransformed
              ? 'After rotating the coordinate system, one horizontal split perfectly captures the diagonal boundary.'
              : 'Each staircase step is a separate axis-aligned split. The tree wastes capacity approximating what is fundamentally a simple line.'}
          </p>
        </div>
      </div>

      <button
        type="button"
        className={`btn ${isTransformed ? 'active' : ''}`}
        onClick={() => setIsTransformed(!isTransformed)}
        style={{ marginTop: '32px' }}
      >
        {isTransformed ? 'Revert to Original' : 'Apply Learned Coordinate'}
      </button>
    </div>
  );
}

// --- VISUALIZATION 2: Ring ---
type RingPoint = { ox: number; oy: number; ring: boolean; ang: number; r: number };

function RingViz() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const animationFrame = useRef<number | null>(null);
  const [isOn, setIsOn] = useState(false);
  const points = useRef<RingPoint[] | null>(null);

  useEffect(() => {
    if (!points.current) {
      const seeded: RingPoint[] = [];
      for (let i = 0; i < 300; i += 1) {
        const angle = Math.random() * Math.PI * 2;
        const isRing = Math.random() > 0.5;
        const noise = (Math.random() - 0.5) * 0.15;
        const radius = isRing ? 0.75 + noise : 0.25 + noise;
        seeded.push({
          ox: Math.cos(angle) * radius,
          oy: Math.sin(angle) * radius,
          ring: isRing,
          ang: angle,
          r: radius,
        });
      }
      points.current = seeded;
    }
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    const pts = points.current;
    if (!canvas || !pts) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const width = 600;
    const height = 300;
    canvas.width = width * 2;
    canvas.height = height * 2;
    ctx.scale(2, 2);

    let progress = isOn ? 1 : 0;
    const target = isOn ? 1 : 0;

    const draw = () => {
      progress += (target - progress) * 0.06;
      ctx.clearRect(0, 0, width, height);

      ctx.save();
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = MUTE;
      if (progress < 0.5) {
        ctx.fillText('x₁', width / 2 + width * 0.38, height / 2 + 14);
        ctx.fillText('x₂', width / 2 + 6, height / 2 - width * 0.36);
      } else {
        ctx.fillText('radius (r)', width / 2 + width * 0.3, height / 2 + 14);
        ctx.fillText('angle (θ)', width / 2 + 6, height / 2 - width * 0.34);
      }
      ctx.restore();

      ctx.strokeStyle = LINE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(width / 2, 10);
      ctx.lineTo(width / 2, height - 10);
      ctx.moveTo(20, height / 2);
      ctx.lineTo(width - 20, height / 2);
      ctx.stroke();

      if (progress > 0.1) {
        ctx.save();
        ctx.globalAlpha = progress;
        ctx.fillStyle = `rgba(213, 43, 30, 0.08)`;
        const separatorX = width / 2 + 0.5 * width * 0.36;
        ctx.fillRect(separatorX, 0, width - separatorX, height);
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(separatorX, 0);
        ctx.lineTo(separatorX, height);
        ctx.stroke();
        ctx.restore();
      }

      if (progress < 0.7) {
        ctx.save();
        ctx.globalAlpha = (1 - progress) * 0.18;
        ctx.strokeStyle = INK;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 0.5 * width * 0.36, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      pts.forEach((point) => {
        const targetX = point.r;
        const targetY = point.ang / (Math.PI * 2);
        const x = point.ox * (1 - progress) + targetX * progress;
        const y = point.oy * (1 - progress) + (targetY * 2 - 1) * progress;
        const px = width / 2 + x * width * 0.36;
        const py = height / 2 + y * height * 0.36;
        ctx.beginPath();
        ctx.arc(px, py, 2.8, 0, Math.PI * 2);
        if (point.ring) {
          ctx.fillStyle = `rgba(14, 14, 14, ${0.55 + progress * 0.25})`;
        } else {
          ctx.fillStyle = `rgba(213, 43, 30, ${0.55 + progress * 0.25})`;
        }
        ctx.fill();
      });

      animationFrame.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      if (animationFrame.current !== null) {
        cancelAnimationFrame(animationFrame.current);
      }
    };
  }, [isOn]);

  return (
    <div className="viz">
      <canvas
        ref={ref}
        style={{
          width: '100%',
          maxWidth: '600px',
          aspectRatio: '600/300',
          display: 'block',
          margin: '0 auto',
          cursor: 'pointer',
        }}
        onClick={() => setIsOn(!isOn)}
      />
      <div
        style={{
          marginTop: '20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '14px',
        }}
      >
        <button type="button" className={`btn ${isOn ? 'active' : ''}`} onClick={() => setIsOn(!isOn)}>
          {isOn ? 'View original (x₁, x₂)' : 'Transform to (radius, angle)'}
        </button>
      </div>
      <div className="stat-strip">
        <div className="stat">
          <div className="idx">[01] // original leaves</div>
          <div className={`val ${isOn ? 'muted' : ''}`}>601</div>
        </div>
        <div className="stat">
          <div className="idx">[02] // transformed leaves</div>
          <div className={`val ${isOn ? 'accent' : 'muted'}`}>28</div>
        </div>
        <div className="stat">
          <div className="idx">[03] // compression</div>
          <div className={`val ${isOn ? 'accent' : 'muted'}`}>21×</div>
        </div>
      </div>
      <p className="viz-cap">
        {isOn
          ? 'In the new coordinate system, the two classes are separated by a single vertical line — one split is enough.'
          : 'In the original (x₁, x₂) space, the boundary is a circle. Trees must approximate circles with hundreds of axis-aligned rectangles.'}
      </p>
    </div>
  );
}

// --- VISUALIZATION 3: Prefix Curve ---
function PrefixCurveViz() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [hoveredMethod, setHoveredMethod] = useState<string | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const width = 600;
    const height = 320;
    canvas.width = width * 2;
    canvas.height = height * 2;
    ctx.scale(2, 2);

    const pad = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const maxLeaves = 650;
    const rawTarget = 0.9962;

    const budgetCurve: [number, number][] = [];
    const learnedCurve: [number, number][] = [];

    for (let leaves = 2; leaves <= maxLeaves; leaves += 4) {
      const budgetAuc = 0.5 + 0.423 * (1 - Math.exp(-leaves / 120));
      budgetCurve.push([leaves, budgetAuc]);
      const learnedAuc = 0.5 + 0.497 * (1 - Math.exp(-leaves / 8));
      learnedCurve.push([leaves, Math.min(learnedAuc, 0.9972)]);
    }

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i += 1) {
      const y = pad.top + (plotH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    const target95 = rawTarget * 0.95;
    const targetY = pad.top + plotH * (1 - (target95 - 0.5) / 0.5);
    ctx.strokeStyle = INK;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, targetY);
    ctx.lineTo(pad.left + plotW, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = MUTE;
    ctx.fillText('95% of raw AUC', pad.left + plotW - 105, targetY - 6);

    const drawCurve = (data: [number, number][], color: string, lineWidth: number, highlight: boolean) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = highlight ? lineWidth + 1 : lineWidth;
      ctx.globalAlpha = highlight ? 1 : 0.85;
      ctx.beginPath();
      data.forEach(([leaves, auc], i) => {
        const x = pad.left + (leaves / maxLeaves) * plotW;
        const y = pad.top + plotH * (1 - (auc - 0.5) / 0.5);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    drawCurve(budgetCurve, MUTE, 2, hoveredMethod === 'budget');
    drawCurve(learnedCurve, ACCENT, 2.5, hoveredMethod === 'learned');

    const learnedTargetLeaves = 15.8;
    const learnedTargetX = pad.left + (learnedTargetLeaves / maxLeaves) * plotW;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(learnedTargetX, targetY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = ACCENT;
    ctx.fillText('15.8 leaves', learnedTargetX + 8, targetY - 8);

    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = MUTE;
    ctx.fillText('never reaches target', pad.left + plotW - 150, pad.top + plotH * (1 - (0.923 - 0.5) / 0.5) - 10);

    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = MUTE;
    ctx.textAlign = 'center';
    ctx.fillText('CUMULATIVE LEAVES', pad.left + plotW / 2, height - 8);
    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('TEST AUC', 0, 0);
    ctx.restore();
    ctx.textAlign = 'left';

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = MUTE;
    ctx.textAlign = 'right';
    for (let v = 0.5; v <= 1.0; v += 0.1) {
      const y = pad.top + plotH * (1 - (v - 0.5) / 0.5);
      ctx.fillText(v.toFixed(1), pad.left - 8, y + 4);
    }

    ctx.textAlign = 'center';
    for (let v = 0; v <= maxLeaves; v += 100) {
      const x = pad.left + (v / maxLeaves) * plotW;
      ctx.fillText(String(v), x, pad.top + plotH + 18);
    }

    ctx.textAlign = 'left';
    const legendX = pad.left + 20;
    const legendY = pad.top + 16;
    ctx.fillStyle = MUTE;
    ctx.fillRect(legendX, legendY - 4, 14, 3);
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = INK_2;
    ctx.fillText('STANDARD XGBOOST (BUDGETED)', legendX + 20, legendY);
    ctx.fillStyle = ACCENT;
    ctx.fillRect(legendX, legendY + 16, 14, 3);
    ctx.fillStyle = INK_2;
    ctx.fillText('WITH LEARNED COORDINATES', legendX + 20, legendY + 20);
  }, [hoveredMethod]);

  return (
    <div className="viz">
      <canvas
        ref={ref}
        style={{
          width: '100%',
          maxWidth: '600px',
          aspectRatio: '600/320',
          display: 'block',
          margin: '0 auto',
        }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - rect.left) / rect.width;
          setHoveredMethod(x > 0.5 ? 'budget' : 'learned');
        }}
        onMouseLeave={() => setHoveredMethod(null)}
      />
      <p className="viz-cap">
        <strong>Ring vs. Center task.</strong> The budgeted tree (grey) plateaus at AUC 0.923 — it can never
        reach 95% of raw AUC within its leaf budget. The learned transform (red) crosses the target in just
        15.8 leaves.
      </p>
    </div>
  );
}

// --- VISUALIZATION 4: Pipeline ---
function PipelineDiagram() {
  const [activePhase, setActivePhase] = useState<0 | 1 | 2>(0);

  const phases = [
    { label: 'Overview', key: 'overview' },
    { label: 'Phase 1 · Probability', key: '01' },
    { label: 'Phase 2 · Teacher', key: '02' },
  ] as const;

  return (
    <div className="viz">
      <div
        style={{
          display: 'flex',
          gap: '8px',
          marginBottom: '28px',
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}
      >
        {phases.map((phase, i) => (
          <button
            key={phase.label}
            type="button"
            className={`btn ${activePhase === i ? 'active' : ''}`}
            onClick={() => setActivePhase(i as 0 | 1 | 2)}
          >
            [{phase.key}] {phase.label}
          </button>
        ))}
      </div>

      <svg
        viewBox="0 0 700 220"
        style={{ width: '100%', maxWidth: '700px', display: 'block', margin: '0 auto' }}
      >
        {/* Phase 1 */}
        <g
          style={{
            opacity: activePhase === 0 || activePhase === 1 ? 1 : 0.25,
            transition: 'opacity 0.3s',
          }}
        >
          <rect x="10" y="20" width="100" height="50" fill="transparent" stroke={LINE} strokeWidth="1.5" />
          <text x="60" y="42" textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono, monospace" fill={INK_2}>
            RAW DATA
          </text>
          <text x="60" y="56" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            (x₁, x₂, ...)
          </text>

          <line x1="110" y1="45" x2="150" y2="45" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow)" />

          <rect
            x="150"
            y="20"
            width="120"
            height="50"
            fill={activePhase === 1 ? INK : 'transparent'}
            stroke={INK}
            strokeWidth="1.5"
          />
          <text
            x="210"
            y="42"
            textAnchor="middle"
            fontSize="10"
            fontFamily="JetBrains Mono, monospace"
            fill={activePhase === 1 ? PAPER_BG_INV : INK}
          >
            PROB. TRANSFORM
          </text>
          <text
            x="210"
            y="56"
            textAnchor="middle"
            fontSize="9"
            fontFamily="JetBrains Mono, monospace"
            fill={activePhase === 1 ? '#999' : MUTE}
          >
            remap + 8 lifts
          </text>

          <line x1="270" y1="45" x2="310" y2="45" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow)" />

          <rect
            x="310"
            y="20"
            width="110"
            height="50"
            fill="transparent"
            stroke={INK}
            strokeWidth="1.5"
          />
          <text x="365" y="42" textAnchor="middle" fontSize="10" fontFamily="JetBrains Mono, monospace" fill={INK}>
            BOOSTER A
          </text>
          <text x="365" y="56" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            8 rounds
          </text>
        </g>

        {/* Connection */}
        <g style={{ opacity: activePhase === 0 ? 1 : 0.3, transition: 'opacity 0.3s' }}>
          <path d="M 365 70 L 365 100 L 520 100 L 520 130" fill="none" stroke={MUTE} strokeWidth="1" strokeDasharray="4 4" />
          <text x="443" y="95" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            base margins
          </text>
        </g>

        {/* Phase 2 */}
        <g
          style={{
            opacity: activePhase === 0 || activePhase === 2 ? 1 : 0.25,
            transition: 'opacity 0.3s',
          }}
        >
          <rect x="10" y="130" width="100" height="50" fill="transparent" stroke={LINE} strokeWidth="1.5" />
          <text x="60" y="152" textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono, monospace" fill={INK_2}>
            RAW DATA
          </text>
          <text x="60" y="166" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            (x₁, x₂, ...)
          </text>

          <line x1="110" y1="155" x2="150" y2="155" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow)" />

          <rect
            x="150"
            y="130"
            width="120"
            height="50"
            fill={activePhase === 2 ? ACCENT : 'transparent'}
            stroke={ACCENT}
            strokeWidth="1.5"
          />
          <text
            x="210"
            y="152"
            textAnchor="middle"
            fontSize="10"
            fontFamily="JetBrains Mono, monospace"
            fill={activePhase === 2 ? PAPER_BG_INV : ACCENT}
          >
            TEACHER TRANSFORM
          </text>
          <text
            x="210"
            y="166"
            textAnchor="middle"
            fontSize="9"
            fontFamily="JetBrains Mono, monospace"
            fill={activePhase === 2 ? '#f5f3ec' : MUTE}
          >
            remap only
          </text>

          <line x1="270" y1="155" x2="310" y2="155" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow)" />

          <rect
            x="310"
            y="130"
            width="110"
            height="50"
            fill="transparent"
            stroke={ACCENT}
            strokeWidth="1.5"
          />
          <text x="365" y="152" textAnchor="middle" fontSize="10" fontFamily="JetBrains Mono, monospace" fill={ACCENT}>
            BOOSTER B
          </text>
          <text x="365" y="166" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            42 rounds
          </text>
        </g>

        {/* Final */}
        <g style={{ opacity: activePhase === 0 ? 1 : 0.4, transition: 'opacity 0.3s' }}>
          <line x1="420" y1="45" x2="460" y2="45" stroke={MUTE} strokeWidth="1" />
          <line x1="420" y1="155" x2="460" y2="155" stroke={MUTE} strokeWidth="1" />
          <path d="M 460 45 L 460 155" fill="none" stroke={MUTE} strokeWidth="1" />
          <line x1="460" y1="100" x2="500" y2="100" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow)" />

          <text x="475" y="80" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill={MUTE}>
            sum
          </text>

          <rect x="500" y="75" width="100" height="50" fill={INK} stroke={INK} strokeWidth="1.5" />
          <text x="550" y="97" textAnchor="middle" fontSize="10" fontFamily="JetBrains Mono, monospace" fill={PAPER_BG_INV}>
            σ(MARGINS)
          </text>
          <text x="550" y="111" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill="#999">
            prediction
          </text>

          <line x1="600" y1="100" x2="640" y2="100" stroke={INK} strokeWidth="1.5" markerEnd="url(#pl-arrow-dark)" />
          <text x="665" y="104" textAnchor="middle" fontSize="14" fontFamily="JetBrains Mono, monospace" fill={INK}>
            ŷ
          </text>
        </g>

        <defs>
          <marker id="pl-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={INK} />
          </marker>
          <marker id="pl-arrow-dark" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={INK} />
          </marker>
        </defs>
      </svg>

      <p className="viz-cap">
        {activePhase === 0
          ? 'The staged recipe chains two independently trained transforms. Phase 1 captures broad geometry; Phase 2 refines numeric details. Their boosting rounds are connected through base margins.'
          : null}
        {activePhase === 1
          ? 'Phase 1 trains a probability-supervised transform that keeps all 8 lift channels. These capture the broad geometric structure of the data — radial patterns, rotations, nonlinear manifolds.'
          : null}
        {activePhase === 2
          ? 'Phase 2 trains a teacher-margin transform that only uses the remapped numeric coordinates (no lift channels). It corrects the fine-grained details that probability training misses. Booster B starts from Booster A\'s margins.'
          : null}
      </p>
    </div>
  );
}

// --- VISUALIZATION 5: Synthetic Results Table ---
function SyntheticResultsTable() {
  const data = [
    {
      task: 'Rotated Hyperplane',
      icon: '↗',
      budgetAuc: 0.9795,
      budgetLeaves: 300.8,
      learnedAuc: 0.9998,
      learnedLeaves: 13.2,
      ratio: '23×',
    },
    {
      task: 'Ring vs. Center',
      icon: '◎',
      budgetAuc: 0.9228,
      budgetLeaves: null,
      learnedAuc: 0.9972,
      learnedLeaves: 15.8,
      ratio: '∞',
    },
    {
      task: 'Tree-Native (control)',
      icon: '▦',
      budgetAuc: 0.9998,
      budgetLeaves: 10.8,
      learnedAuc: 0.9998,
      learnedLeaves: 10.8,
      ratio: '1×',
    },
  ];

  return (
    <div className="table-wrap">
      <table className="v9-table">
        <thead>
          <tr className="head">
            <th className="ink">Task</th>
            <th className="center" colSpan={2}>
              Standard XGBoost
            </th>
            <th className="center accent" colSpan={2}>
              Learned Coordinates
            </th>
            <th className="center ink">Compression</th>
          </tr>
          <tr className="subhead">
            <th />
            <th className="center">AUC</th>
            <th className="center">L₉₅</th>
            <th className="center">AUC</th>
            <th className="center">L₉₅</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.task}>
              <td className="label">
                <span className="icon">{row.icon}</span>
                {row.task}
              </td>
              <td className="center">{row.budgetAuc.toFixed(4)}</td>
              <td className={`center ${row.budgetLeaves === null ? 'never' : ''}`}>
                {row.budgetLeaves === null ? 'never' : row.budgetLeaves.toFixed(1)}
              </td>
              <td className="center accent">{row.learnedAuc.toFixed(4)}</td>
              <td className="center accent">{row.learnedLeaves.toFixed(1)}</td>
              <td className="center">
                <span className="ratio" style={{ color: row.ratio === '1×' ? MUTE : INK }}>
                  {row.ratio}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="viz-cap" style={{ textAlign: 'left', marginTop: '12px' }}>
        <Code>L₉₅</Code> = cumulative leaves needed to reach 95% of raw (unconstrained) AUC. Lower is better.
        The control task confirms the transform does not hurt when the data is already tree-friendly.
      </p>
    </div>
  );
}

// --- VISUALIZATION 6: Benchmark Table ---
function BenchmarkTable() {
  const data = [
    { dataset: 'MagicTelescope', rawAuc: 0.938, budget: 588.4, prob: 70.0, staged: 70.0 },
    { dataset: 'PhonemeSpectra', rawAuc: 0.9551, budget: 612.5, prob: 318.5, staged: 279.0 },
    { dataset: 'vehicle', rawAuc: 0.9985, budget: 51.0, prob: 14.5, staged: 16.4 },
    { dataset: 'spambase', rawAuc: 0.989, budget: 188.8, prob: 138.8, staged: 101.6 },
    { dataset: 'ailerons', rawAuc: 0.9567, budget: 134.4, prob: 25.6, staged: 25.6 },
    { dataset: 'puma32H', rawAuc: 0.9576, budget: 41.6, prob: 44.8, staged: 44.8 },
    { dataset: 'cpu_act', rawAuc: 0.9838, budget: 130.4, prob: 37.4, staged: 37.4 },
  ];

  return (
    <div className="table-wrap">
      <table className="v9-table">
        <thead>
          <tr className="head">
            <th className="ink">Dataset</th>
            <th className="center">Raw AUC</th>
            <th className="center">Budget L₉₈</th>
            <th className="center">Prob. L₉₈</th>
            <th className="center accent">Staged L₉₈</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const min = Math.min(row.budget, row.prob, row.staged);
            return (
              <tr key={row.dataset}>
                <td className="label">{row.dataset}</td>
                <td className="center">{row.rawAuc.toFixed(4)}</td>
                <td className={`center ${row.budget === min ? 'ink' : ''}`}>{row.budget.toFixed(1)}</td>
                <td className={`center ${row.prob === min ? 'ink' : ''}`}>{row.prob.toFixed(1)}</td>
                <td className={`center ${row.staged === min ? 'accent' : ''}`}>{row.staged.toFixed(1)}</td>
              </tr>
            );
          })}
          <tr className="summary">
            <td>Avg. Rank</td>
            <td className="center">—</td>
            <td className="center">3.36</td>
            <td className="center">1.86</td>
            <td className="center accent">1.71</td>
          </tr>
        </tbody>
      </table>
      <p className="viz-cap" style={{ textAlign: 'left', marginTop: '12px' }}>
        <Code>L₉₈</Code> = cumulative leaves to reach 98% of raw AUC (lower is better). The staged recipe
        achieves the best average rank across all 7 datasets. Bold = best per row.
      </p>
    </div>
  );
}

// --- VISUALIZATION 7: Gain Share ---
function GainShareViz() {
  const datasets = [
    { name: 'MagicTelescope', probLift: 0.849, probRemap: 0.151, teacherLift: 0.248, teacherRemap: 0.752 },
    { name: 'PhonemeSpectra', probLift: 0.571, probRemap: 0.429, teacherLift: 0.229, teacherRemap: 0.771 },
    { name: 'vehicle', probLift: 0.615, probRemap: 0.385, teacherLift: 0.024, teacherRemap: 0.976 },
    { name: 'spambase', probLift: 0.012, probRemap: 0.988, teacherLift: 0.007, teacherRemap: 0.993 },
  ];

  return (
    <div className="gain">
      <div className="col">
        <h5 className="ink">// Probability Supervision</h5>
        {datasets.map((d) => (
          <div key={`${d.name}-prob`} className="row">
            <div className="name">{d.name}</div>
            <div className="bar">
              <div
                className="seg"
                style={{
                  width: `${d.probRemap * 100}%`,
                  background: 'rgba(14, 14, 14, 0.18)',
                  color: INK,
                }}
              >
                {d.probRemap > 0.15 ? `${(d.probRemap * 100).toFixed(0)}%` : null}
              </div>
              <div
                className="seg"
                style={{
                  width: `${d.probLift * 100}%`,
                  background: INK,
                  color: PAPER_BG_INV,
                }}
              >
                {d.probLift > 0.15 ? `${(d.probLift * 100).toFixed(0)}%` : null}
              </div>
            </div>
          </div>
        ))}
        <div className="legend">
          <span>
            <span className="swatch" style={{ background: 'rgba(14, 14, 14, 0.18)' }} />
            remap
          </span>
          <span>
            <span className="swatch" style={{ background: INK }} />
            lift
          </span>
        </div>
      </div>

      <div className="col">
        <h5 className="accent">// Teacher Supervision</h5>
        {datasets.map((d) => (
          <div key={`${d.name}-teacher`} className="row">
            <div className="name">{d.name}</div>
            <div className="bar">
              <div
                className="seg"
                style={{
                  width: `${d.teacherRemap * 100}%`,
                  background: ACCENT,
                  color: PAPER_BG_INV,
                }}
              >
                {d.teacherRemap > 0.15 ? `${(d.teacherRemap * 100).toFixed(0)}%` : null}
              </div>
              <div
                className="seg"
                style={{
                  width: `${d.teacherLift * 100}%`,
                  background: 'rgba(213, 43, 30, 0.2)',
                  color: ACCENT,
                }}
              >
                {d.teacherLift > 0.15 ? `${(d.teacherLift * 100).toFixed(0)}%` : null}
              </div>
            </div>
          </div>
        ))}
        <div className="legend">
          <span>
            <span className="swatch" style={{ background: ACCENT }} />
            remap
          </span>
          <span>
            <span className="swatch" style={{ background: 'rgba(213, 43, 30, 0.2)' }} />
            lift
          </span>
        </div>
      </div>

      <p className="viz-cap" style={{ gridColumn: '1 / -1', marginTop: '12px' }}>
        <strong>How trees use the transformed features.</strong> Probability supervision preserves lift
        channels (the dark bars on the left), especially for geometric datasets like MagicTelescope. Teacher
        supervision concentrates almost all gain in the remap — it suppresses lift channels. This asymmetry
        is why the two phases complement each other.
      </p>
    </div>
  );
}

// --- VISUALIZATION 8: Alternating Loop ---
function AlternatingLoopViz() {
  const [step, setStep] = useState(0);
  const steps = [
    { label: 'Transform Data', desc: 'Apply the current neural transform to get new coordinates.' },
    { label: 'Fit XGBoost', desc: 'Train a small gradient boosted tree on the transformed data.' },
    { label: 'Train Surrogate', desc: 'Fit a differentiable MLP to mimic the tree\'s predictions.' },
    { label: 'Update Transform', desc: 'Backpropagate through the surrogate to improve the transform.' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => (s + 1) % 4);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="viz">
      <svg viewBox="0 0 400 220" style={{ width: '100%', maxWidth: '420px', display: 'block', margin: '0 auto' }}>
        {steps.map((s, i) => {
          const angle = (i / 4) * Math.PI * 2 - Math.PI / 2;
          const cx = 200 + Math.cos(angle) * 80;
          const cy = 110 + Math.sin(angle) * 70;
          const isActive = step === i;

          const nextAngle = ((i + 1) / 4) * Math.PI * 2 - Math.PI / 2;
          const nextCx = 200 + Math.cos(nextAngle) * 80;
          const nextCy = 110 + Math.sin(nextAngle) * 70;

          return (
            <g key={s.label}>
              <line
                x1={cx + (nextCx - cx) * 0.3}
                y1={cy + (nextCy - cy) * 0.3}
                x2={nextCx - (nextCx - cx) * 0.3}
                y2={nextCy - (nextCy - cy) * 0.3}
                stroke={isActive ? ACCENT : LINE}
                strokeWidth={isActive ? 2 : 1}
                markerEnd={isActive ? 'url(#loop-arrow-active)' : 'url(#loop-arrow-gray)'}
                style={{ transition: 'all 0.3s' }}
              />
              <circle
                cx={cx}
                cy={cy}
                r={isActive ? 30 : 26}
                fill={isActive ? ACCENT : 'transparent'}
                stroke={isActive ? ACCENT : INK}
                strokeWidth={isActive ? 2.5 : 1.5}
                style={{ transition: 'all 0.3s' }}
              />
              <text
                x={cx}
                y={cy + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="18"
                fontFamily="Helvetica Neue, Inter Tight, Arial, sans-serif"
                fontWeight="700"
                fill={isActive ? PAPER_BG_INV : INK}
                style={{ transition: 'fill 0.3s' }}
              >
                {i + 1}
              </text>
            </g>
          );
        })}
        <defs>
          <marker id="loop-arrow-active" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill={ACCENT} />
          </marker>
          <marker id="loop-arrow-gray" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill={LINE} />
          </marker>
        </defs>
      </svg>

      <div style={{ textAlign: 'center', marginTop: '20px', minHeight: '60px' }}>
        <div
          style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            fontWeight: 500,
            color: ACCENT,
            textTransform: 'uppercase',
            letterSpacing: '0.16em',
            marginBottom: '8px',
          }}
        >
          [{String(step + 1).padStart(2, '0')}] // {steps[step].label}
        </div>
        <p
          style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '13px',
            color: INK_2,
            maxWidth: '420px',
            margin: '0 auto',
            lineHeight: 1.6,
          }}
        >
          {steps[step].desc}
        </p>
      </div>

      <p className="viz-cap">
        This loop repeats 15 times. The surrogate MLP is the key trick — since XGBoost isn't differentiable,
        the surrogate acts as a smooth approximation that gradients can flow through to update the
        coordinate transform.
      </p>
    </div>
  );
}

// --- MAIN COMPONENT ---
export default function SLCPaper() {
  return (
    <div className="v9">
      <div className="pad">
        {/* Topline */}
        <div className="topline">
          <div className="brand">
            <SlashMark size={22} />
            <span>Inductive.ML</span>
          </div>
          <div className="meta">Independent AI Research Lab — Bangalore, India — Est. MMXXV</div>
          <div className="links">
            <a href="/">← Index</a>
          </div>
        </div>

        <div className="grid12">
          {/* Crumb */}
          <nav className="crumb" aria-label="Breadcrumb">
            <a href="/">Inductive.ML</a>
            <span className="sep">/</span>
            <a href="/#research">Research</a>
            <span className="sep">/</span>
            <span className="here">Staged Learned Coordinates</span>
          </nav>

          {/* Hero */}
          <Secbar num="00" ttl="research-explainer" meta="2026 · preprint" />

          <div className="paper-hero">
            <div className="head">
              <h1>Teaching Trees to Understand Geometry</h1>
              <p className="sub">
                A plain-English breakdown of our paper:{' '}
                <em>“Staged Learned Coordinates for Gradient Boosted Trees on Continuous Tabular Data.”</em>
              </p>
              <div className="links">
                <a
                  className="btn"
                  href="https://github.com/inductiveML/staged-learned-coordinates"
                  target="_blank"
                  rel="noreferrer"
                >
                  View Source ↗
                </a>
              </div>
            </div>
            <div className="meta-rail">
              <div className="row">
                <span>// code</span>
                <span>SLC</span>
              </div>
              <div className="row">
                <span>// status</span>
                <span>preprint</span>
              </div>
              <div className="row">
                <span>// year</span>
                <span>2026</span>
              </div>
              <div className="row">
                <span>// task</span>
                <span>tabular</span>
              </div>
              <div className="row">
                <span>// stack</span>
                <span>xgboost</span>
              </div>
            </div>
          </div>

          <Callout label="tl;dr">
            Decision trees are powerful on tabular data, but they waste capacity when the data has curved or
            rotated geometry. We put a small neural network <strong>in front</strong> of XGBoost that learns
            to straighten the geometry before the tree sees it. Result: a problem that required 601 tree
            leaves now needs only 28 — a <strong>21× compression</strong> — with no loss in accuracy.
          </Callout>

          {/* §01 Background */}
          <Secbar num="01" ttl="background" meta="why trees" />
          <div className="paper">
            <p>
              If you've worked with structured data — spreadsheets, databases, CSVs — you've probably heard
              of <strong>XGBoost</strong>. Gradient Boosted Trees (GBTs) have won more Kaggle competitions on
              tabular data than any other model family. They handle missing values gracefully, don't need
              feature scaling, and are remarkably robust to hyperparameter choices.
            </p>
            <p>
              But why? The intuition is simple: a tree makes <em>axis-aligned splits</em>. It looks at one
              column at a time and asks: "is age &gt; 35?" or "is income &lt; 50,000?". Then it builds up
              complex decisions by stacking hundreds of these simple yes/no questions.
            </p>
            <p>
              This works beautifully when the important patterns in your data actually align with the
              columns you measured. But what happens when they don't?
            </p>
          </div>

          {/* §02 Blind Spot */}
          <Secbar num="02" ttl="the blind spot" meta="diagonal boundaries" />
          <div className="paper">
            <p>
              Imagine you're trying to classify points as "above" or "below" a diagonal line. The true
              boundary runs at 45 degrees, but a tree can only make horizontal and vertical cuts. To
              approximate the diagonal, it has to build a <strong>staircase</strong> — many tiny rectangular
              bites that zigzag along the true line.
            </p>
            <p>
              Each step of the staircase is a separate tree leaf. More leaves means a bigger, slower model
              that's more likely to overfit. The tree isn't wrong — it will eventually get there — but it's
              doing an enormous amount of work to express something that's fundamentally simple.
            </p>
          </div>

          <StaircaseViz />

          <div className="paper">
            <p>
              This is the core tension. The data has a <em>geometry</em> — a natural shape to its decision
              boundary — and the tree has a <em>coordinate system</em> it's stuck with. When they don't
              align, the tree wastes leaves.
            </p>
          </div>

          <Callout label="key question">
            Instead of measuring final accuracy, measure <strong>how many leaves</strong> a tree needs to
            reach a target fraction of its best possible accuracy. This is the <strong>Lτ</strong> metric —
            target-prefix leaves.
          </Callout>

          {/* §03 The Idea */}
          <Secbar num="03" ttl="the idea" meta="rotate before cutting" />
          <div className="paper">
            <p>
              What if we could <strong>change the coordinate system</strong> before handing data to the
              tree? If the boundary is diagonal, rotate the data so it becomes horizontal. If it's circular,
              map it to polar coordinates so it becomes a straight line.
            </p>
            <p>
              That's exactly what we do. We place a small neural network (a 3-layer MLP with 128 hidden
              units) <em>in front of</em> XGBoost. The network's job isn't to make predictions — it's to{' '}
              <strong>transform the coordinates</strong> so that the tree's axis-aligned splits become
              maximally efficient.
            </p>
            <p>
              Think of it like a translator. The neural network translates from the language the data speaks
              (circles, diagonals, curved manifolds) into the language the tree speaks (axis-aligned
              rectangles).
            </p>
          </div>

          <div className="three-up">
            {[
              {
                icon: 'f(x)',
                title: 'Remap',
                desc: 'A learned linear + nonlinear transform that replaces the original numeric features. Initialized as identity, so it starts neutral.',
              },
              {
                icon: '+8',
                title: 'Lift Channels',
                desc: '8 additional features the network invents from scratch. These can encode geometry (like radial distance) that no single original column captures.',
              },
              {
                icon: 'Q⊤Q≈I',
                title: 'Orthogonality',
                desc: 'A regularizer that keeps the remap matrix well-conditioned, preventing it from collapsing dimensions.',
              },
            ].map((item) => (
              <div className="cell" key={item.title}>
                <div className="icon">{item.icon}</div>
                <h4>{item.title}</h4>
                <p>{item.desc}</p>
              </div>
            ))}
          </div>

          {/* §04 Ring */}
          <Secbar num="04" ttl="the 21× compression" meta="rings · radial distance" />
          <div className="paper">
            <p>
              Let's see this in action with a classic problem: <strong>ring vs. center</strong>. One class
              forms a ring; the other clusters in the center. The boundary is a circle — about the worst
              possible shape for axis-aligned splits.
            </p>
            <p>
              A standard budgeted XGBoost needs <strong>601 leaves</strong> and still only reaches an AUC of
              0.923 — it <em>never</em> gets within 95% of the unconstrained baseline within its leaf
              budget.
            </p>
            <p>
              But when we apply the learned coordinate transform, something remarkable happens. Without
              being told anything about circles or radial distance, the neural network independently{' '}
              <strong>discovers</strong> that the optimal remapping is to convert from (x₁, x₂) to (radius,
              angle). In this new coordinate system, the circular boundary becomes a vertical line — one
              split.
            </p>
          </div>

          <RingViz />

          <div className="paper">
            <p>
              The transformed tree reaches the same accuracy in <strong>28 leaves</strong>. That's a 21×
              compression. The network didn't just help the tree — it found the geometry that the problem
              was hiding.
            </p>
          </div>

          {/* §05 Prefix Curve */}
          <Secbar num="05" ttl="the prefix curve" meta="efficiency, not accuracy" />
          <div className="paper">
            <p>
              The paper's headline metric isn't accuracy — it's <strong>efficiency</strong>. We plot the
              test AUC as a function of cumulative tree leaves (the "prefix curve"). This tells you: at any
              given model size, how good is your prediction?
            </p>
            <p>
              The question we're answering is:{' '}
              <em>"How many leaves do you need before you're within τ% of the best unrestricted tree?"</em>{' '}
              We call this <Code>Lτ</Code> — the target-prefix leaves. Lower is better.
            </p>
          </div>

          <PrefixCurveViz />

          <div className="paper">
            <p>
              We also measure <Code>A≤128</Code>, the area under the prefix curve clipped at 128 leaves.
              This rewards methods that advance performance <em>early</em> — important for deployment
              scenarios where model size is constrained.
            </p>
          </div>

          {/* §06 Synthetic */}
          <Secbar num="06" ttl="proof of mechanism" meta="three synthetic tasks" />
          <div className="paper">
            <p>
              Before touching real data, we verified the mechanism on three carefully designed tasks. Each
              one tests a different failure mode:
            </p>
          </div>

          <div className="task-list">
            {[
              {
                icon: '↗',
                name: 'Rotated Sparse Hyperplane',
                desc: 'A simple linear boundary in a randomly rotated space. Tests whether the transform can undo arbitrary rotations.',
                result: '300.8 → 13.2 leaves (23× compression)',
              },
              {
                icon: '◎',
                name: 'Ring vs. Center',
                desc: 'A nonlinear radial geometry. Tests whether the transform can discover and exploit curved structure.',
                result: 'Budgeted tree never reaches target. Transform: 15.8 leaves.',
              },
              {
                icon: '▦',
                name: 'Tree-Native Piecewise (control)',
                desc: "An axis-aligned task that's already perfect for trees. Tests that the transform doesn't hurt when it's not needed.",
                result: '10.8 → 10.8 leaves (no change — correct!)',
              },
            ].map((task) => (
              <div className="row" key={task.name}>
                <div className="icon">{task.icon}</div>
                <div className="body">
                  <div className="name">{task.name}</div>
                  <div className="desc">{task.desc}</div>
                  <div className="result">{task.result}</div>
                </div>
              </div>
            ))}
          </div>

          <SyntheticResultsTable />

          <div className="paper">
            <p>
              The control task is critical. If the transform improved results on tasks where trees are
              already optimal, it would mean we're just adding raw model capacity, not actually improving
              the coordinate system. The fact that both methods tie at 10.8 leaves confirms the mechanism is
              geometric, not just parametric.
            </p>
          </div>

          {/* §07 Training Challenge */}
          <Secbar num="07" ttl="the training challenge" meta="trees aren't differentiable" />
          <div className="paper">
            <p>
              There's a fundamental problem with putting a neural network in front of a tree: you can't
              backpropagate through XGBoost. Neural networks learn through gradients — you compute "if I
              nudge this weight slightly, how does the loss change?" — but XGBoost makes discrete,
              combinatorial split decisions. There's no gradient to follow.
            </p>
            <p>
              Our solution is an <strong>alternating optimization loop</strong> that uses a surrogate MLP as
              a differentiable stand-in for the tree:
            </p>
          </div>

          <AlternatingLoopViz />

          <div className="paper">
            <p>
              The surrogate is never used for prediction. It's purely a training device — a smooth
              approximation of the tree that lets gradients flow back to the coordinate transform. This loop
              repeats 15 times, with 25 surrogate training epochs and 10 transform update epochs per round.
            </p>
          </div>

          {/* §08 Two Views */}
          <Secbar num="08" ttl="two supervision views" meta="probability vs. teacher" />
          <div className="paper">
            <p>
              What should the surrogate try to imitate? We found that the answer depends on what you care
              about. There are two natural targets, and they produce <em>different</em> transforms with
              different strengths:
            </p>
          </div>

          <div className="two-up">
            <div className="cell">
              <div className="lbl">// probability view</div>
              <p>
                The surrogate fits the teacher's predicted class probabilities using binary cross-entropy
                loss.
              </p>
              <div className="meta">
                <strong>Strength:</strong> Preserves lift channels. Captures broad geometric structure
                early.
                <br />
                <strong>Weakness:</strong> May lack fine numeric fidelity in later rounds.
              </div>
            </div>
            <div className="cell">
              <div className="lbl accent">// teacher-margin view</div>
              <p>
                The surrogate fits the raw margin scores from the final boosting round using MSE + BCE
                loss.
              </p>
              <div className="meta">
                <strong>Strength:</strong> Pushes toward late numeric fidelity. Better at correcting
                residual errors.
                <br />
                <strong>Weakness:</strong> Suppresses lift channels. Can overdrive transform.
              </div>
            </div>
          </div>

          <div className="paper">
            <p>
              Neither view dominates the other on all datasets. Probability wins on "lift-heavy" geometric
              datasets (MagicTelescope, ailerons); teacher-margin wins on datasets where the geometry is
              already well-aligned (puma32H). This motivated the staged approach.
            </p>
          </div>

          <GainShareViz />

          {/* §09 Append Trap */}
          <Secbar num="09" ttl="a structural fix" meta="remap, don't append" />
          <div className="paper">
            <p>
              Before we found the staged recipe, we tried a simpler approach: just <strong>append</strong>{' '}
              the neural network's outputs as extra columns alongside the original features. This works on
              synthetic data, but fails on real datasets. Why?
            </p>
          </div>

          <Callout label="the append trap">
            When you append learned features to the original columns, the tree can — and does —{' '}
            <strong>ignore them</strong>. The original features are "easier" to split on because the tree
            has already built intuitions about their ranges. The learned features sit unused, gaining no
            splits, improving nothing.
          </Callout>

          <div className="paper">
            <p>
              The fix is surgical: <strong>replace</strong> the numeric block with the learned remap. Don't
              give the tree a choice between old and new coordinates — force it to work in the new space.
              This is the structural change that makes the approach work on real data.
            </p>
            <p>
              The remap is initialized as the identity matrix, so at the start of training, the tree sees
              exactly the original data. The transform has to <em>earn</em> any deviation from identity by
              actually improving the tree's efficiency.
            </p>
          </div>

          {/* §10 Staged Recipe */}
          <Secbar num="10" ttl="the staged recipe" meta="probability → teacher" />
          <div className="paper">
            <p>
              Since probability and teacher views have complementary strengths, the natural question is: can
              we combine them? We tried several approaches — scalar loss interpolation, two-view refinement,
              branch unions — and they all failed or regressed.
            </p>
            <p>
              The approach that works is deceptively simple: <strong>don't mix them at all</strong>. Train
              each view independently, then chain their boosting rounds in sequence.
            </p>
          </div>

          <PipelineDiagram />

          <div className="callout" style={{ gridColumn: '1 / -1' }}>
            <div className="lbl">// algorithm 01</div>
            <div style={{ alignSelf: 'start' }}>
              <p
                style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '13px',
                  lineHeight: 1.85,
                  color: PAPER_BG_INV,
                  margin: 0,
                  fontWeight: 400,
                  letterSpacing: 0,
                }}
              >
                <strong style={{ color: ACCENT, fontWeight: 500 }}>Staged Probability-to-Teacher Remap</strong>
                <br />
                <br />
                01 · Train a probability-supervised transform. Freeze it. Keep remap + 8 lift channels.
                <br />
                02 · Train a teacher-margin-supervised transform. Freeze it. Keep only remapped numerics.
                <br />
                03 · Fit Booster A on the Phase 1 view for the first T₁ = 8 rounds.
                <br />
                04 · Compute Booster A's raw margins on the training set.
                <br />
                05 · Fit Booster B on the Phase 2 view for the remaining T − T₁ = 42 rounds, using A's
                margins as base margins.
                <br />
                06 · At test time, sum both boosters' margins and apply sigmoid.
              </p>
            </div>
          </div>

          <div className="paper">
            <p>
              The insight: Phase 2 doesn't need lift channels because the broad geometric work is already
              done by Phase 1. Phase 2 only needs the remapped numerics to correct the fine details. This
              division of labor is why the staged recipe outperforms either view alone.
            </p>
          </div>

          {/* §11 Real Results */}
          <Secbar num="11" ttl="real benchmark results" meta="seven datasets" />
          <div className="paper">
            <p>
              We tested on seven continuous-feature binary classification datasets from OpenML. The headline
              metric is <Code>L₉₈</Code> — how many cumulative leaves does each method need to reach 98% of
              the unconstrained raw XGBoost AUC?
            </p>
          </div>

          <BenchmarkTable />

          <div className="paper">
            <p>
              The staged recipe achieves the best average rank (1.71 vs 1.86 for probability alone and 3.36
              for budget). Its biggest wins are on datasets with strong geometric misalignment: spambase
              drops from 188.8 to 101.6 leaves, PhonemeSpectra from 612.5 to 279.0.
            </p>
            <p>
              The one dataset where the budget baseline wins (puma32H at 41.6 vs 44.8) is a case where the
              raw features are already well-aligned with tree splits — the transform has nothing to fix, and
              the small overhead from identity initialization costs a few extra leaves.
            </p>
          </div>

          {/* §12 Held-out */}
          <Secbar num="12" ttl="held-out validation" meta="frozen pre-registered" />
          <div className="paper">
            <p>
              To guard against overfitting our method to the benchmark, we pre-registered a{' '}
              <strong>frozen held-out pack</strong> of four datasets (bank32nh, jm1, kc1, musk) that were
              never used during development. The recipe and all hyperparameters were locked before running
              these experiments.
            </p>
          </div>

          <div className="holdout">
            {[
              { name: 'bank32nh', budget: '201.2', staged: '162.0', verdict: 'win' },
              { name: 'jm1', budget: '19.2', staged: '19.0', verdict: 'tie' },
              { name: 'kc1', budget: '107.2', staged: '14.5', verdict: 'win' },
              { name: 'musk', budget: '2.0', staged: '2.0', verdict: 'tie' },
            ].map((d) => (
              <div className="cell" key={d.name}>
                <div className="name">// {d.name}</div>
                <div className={`num ${d.verdict}`}>{d.staged}</div>
                <div className="was">was {d.budget}</div>
              </div>
            ))}
          </div>

          <div className="paper">
            <p>
              The staged recipe <strong>matches or improves</strong> the budget baseline on all four
              held-out datasets, with no catastrophic regressions. The kc1 result (107.2 → 14.5) was the
              most dramatic improvement — a 7.4× compression on a completely unseen dataset.
            </p>
          </div>

          {/* §13 What Failed */}
          <Secbar num="13" ttl="what didn't work" meta="negative results" />
          <div className="paper">
            <p>
              Research papers often hide their failures. We think the negative results are just as
              instructive:
            </p>
          </div>

          <div className="fail-list">
            {[
              {
                name: 'Scalar Loss Gating',
                desc: 'Interpolating between probability and teacher losses during training. Degraded all tested datasets — MagicTelescope went from 70.0 to 169.2 leaves. The two objectives fight each other when mixed in a single optimization.',
              },
              {
                name: 'Two-View Refinement',
                desc: 'Adding a second round of surrogate training that tries to combine both views. Added complexity without improving the key failure cases. Average rank worsened from 1.33 to 1.67.',
              },
              {
                name: 'Teacher-Heavy Schedules',
                desc: 'Spending most boosting rounds on teacher supervision. Retains good L₉₈ on some datasets but loses up to 0.015 AUC on others. The teacher transform over-specializes the remap at the cost of final accuracy.',
              },
              {
                name: 'Branch Union (meta-baseline)',
                desc: 'Taking the best of both views per dataset. Strongest results, but the union is overwhelmingly probability-dominant (97.7% on MagicTelescope). Not deployable as a single recipe — it requires knowing which view is better per dataset.',
              },
            ].map((item) => (
              <div className="row" key={item.name}>
                <div className="name">// {item.name}</div>
                <p className="desc">{item.desc}</p>
              </div>
            ))}
          </div>

          {/* §14 Limitations */}
          <Secbar num="14" ttl="limitations" meta="open questions" />
          <div className="paper">
            <p>We're deliberately honest about where the approach falls short:</p>
            <ul>
              <li>
                <strong>Continuous data only.</strong> The gains on mixed categorical-numerical datasets
                (adult, credit-g) are smaller and less uniform. Heterogeneous tabular data needs stronger
                locality or type-aware handling.
              </li>
              <li>
                <strong>No statistical significance tests.</strong> We report mean metrics over 5 seeds but
                not confidence intervals. The benchmark is small enough that individual dataset variance
                matters.
              </li>
              <li>
                <strong>Binary classification only.</strong> We haven't tested on regression or multi-class
                tasks.
              </li>
              <li>
                <strong>Training cost.</strong> The alternating loop adds ~15 XGBoost fits and ~375 epochs of
                neural network training on top of a single XGBoost run. For small datasets this is
                negligible; for large ones it may not be.
              </li>
            </ul>
          </div>

          {/* §15 Big Picture */}
          <Secbar num="15" ttl="the big picture" meta="closing" />
          <div className="paper">
            <p>
              This paper is about one specific claim:{' '}
              <strong>
                for continuous binary tabular classification, a staged probability-to-teacher remap recipe
                is the most robust single-recipe approach
              </strong>{' '}
              for compressing tree complexity without sacrificing accuracy.
            </p>
            <p>
              But we think the broader lesson is more interesting. Every model has a geometry — a coordinate
              system where its complexity collapses. Trees want axis-aligned boundaries. Neural networks
              want smooth manifolds. The question isn't "which model is better?" but "what coordinate
              system makes your model see the problem clearly?"
            </p>
            <p>
              We reduced 601 leaves to 28 on a ring problem. Not by building a better tree, or a bigger
              network, but by finding the coordinate system where the tree's natural language —
              axis-aligned splits — is exactly the right language for the problem.
            </p>
          </div>

          <Callout label="closing">
            We didn't make a smarter tree. <strong>We gave it a clearer view of the world.</strong>
          </Callout>

          {/* §16 Cite */}
          <Secbar num="16" ttl="cite" meta="bibtex" />
          <div className="bib">
            <div className="lbl">// cite this work</div>
            <pre>
{`@article{parmar2026staged,
  title  = {Staged Learned Coordinates for Gradient Boosted Trees
            on Continuous Tabular Data},
  author = {Parmar, Aditya Veer},
  year   = {2026}
}`}
            </pre>
          </div>

          {/* Footer */}
          <div className="foot">
            <div className="a">© MMXXVI · Inductive.ML</div>
            <div className="b">Bangalore — India</div>
            <div className="c">
              <a href="/">← Back to Index</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
