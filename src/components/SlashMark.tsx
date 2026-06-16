interface SlashMarkProps {
  size?: number;
  /** Colour of the slash blade. Defaults to the inherited text colour. */
  color?: string;
  /** Colour of the spark node. Defaults to the slash colour. */
  spark?: string;
}

/**
 * Inductive ML brand mark — a terminal "slash" blade with an ember spark node.
 * Drawn on a transparent ground so it reads over the lava surface; colour is
 * inherited from `currentColor` unless overridden.
 */
export default function SlashMark({ size = 24, color = 'currentColor', spark }: SlashMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      style={{ display: 'block', flexShrink: 0 }}
    >
      <line x1="34" y1="74" x2="60" y2="28" stroke={color} strokeWidth="10" strokeLinecap="round" />
      <circle cx="60" cy="28" r="7.5" fill={spark ?? color} />
    </svg>
  );
}
