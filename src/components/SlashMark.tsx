interface SlashMarkProps {
  size?: number;
  bg?: string;
  fg?: string;
  accent?: string;
}

export default function SlashMark({
  size = 28,
  bg = '#0e0e0e',
  fg = '#fafaf7',
  accent = '#d52b1e',
}: SlashMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block', flexShrink: 0 }}
    >
      <rect width="100" height="100" fill={bg} />
      <line x1="22" y1="78" x2="62" y2="22" stroke={fg} strokeWidth="14" strokeLinecap="square" />
      <rect x="68" y="60" width="22" height="22" fill={accent} />
    </svg>
  );
}
