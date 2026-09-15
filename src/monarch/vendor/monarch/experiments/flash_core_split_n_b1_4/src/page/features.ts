// The device features this unit may request. Narrowing a configured string through this tuple is
// what keeps every requestDevice descriptor typed without a cast; an unregistered name throws.
export const REQUESTABLE_FEATURES = ["timestamp-query", "shader-f16", "subgroups"] as const;

export type RequestableFeature = (typeof REQUESTABLE_FEATURES)[number];

export function isRequestableFeature(name: string): name is RequestableFeature {
  const known: readonly string[] = REQUESTABLE_FEATURES;
  return known.includes(name);
}

export function requestableFeature(name: string): RequestableFeature {
  if (!isRequestableFeature(name)) {
    throw new Error(`${name} is not a registered requestable feature`);
  }
  return name;
}
