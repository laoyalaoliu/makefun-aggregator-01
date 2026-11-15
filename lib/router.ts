// Routing: which free model serves a request, and in what order the rest are tried.
//
// Quality is 60% of the score and is the served model's rating over the best rating a
// free model meeting the request's requirements could have earned, so the plan is
// greedy: every free candidate that meets the requirements, best rating first. A
// provider's per-trial budget and failure schedule are never published, so the plan
// does not guess them: when a call refuses (402, 429, a 5xx, or a body this wire cannot
// use), the serve loop simply walks to the next candidate, and the refusal costs
// nothing but the call itself. Two refinements ride under the rating, never above it:
//
//   - a request the client sent in JSON mode is never planned onto the Anthropic wire
//     while another wire can carry it: that wire's parser reports json_mode false
//     whatever it is sent, so such a candidate is ordered after every other wire's
//     candidates of any rating (it stays in the plan as a last resort);
//   - at equal rating, the client's own wire goes first (a pass-through cannot lose a
//     field in translation), then the provider the entry has called least so far
//     (budgets are per provider, so spreading calls keeps more of the plan alive),
//     then the brief's provider order and the model id, so the plan is a pure function
//     of the brief, the requests and the answers seen so far: a replay walks it
//     identically.

import { capabilitiesOf, meetsRequirements, modelFree, ratingOfModel, type Probe, type ProbedModel, type RatingsFile, type Requirement } from './providers.ts';

export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

export interface PlanContext {
  /** The wire the client used: 'openai' or 'anthropic'. */
  clientWire: 'openai' | 'anthropic';
  /** Whether the client asked for JSON mode (only an OpenAI client can). */
  clientJsonMode: boolean;
  /** Provider calls this entry has made so far, by provider id. */
  callsMade: ReadonlyMap<string, number>;
  /**
   * Providers that have said their budget is spent or that they are out: their own
   * refusal messages name both, and neither answer changes for the rest of the trial,
   * so they leave the plan (unless that would leave it empty).
   */
  exhausted?: ReadonlySet<string>;
}

/** Every free model meeting the requirements, in the order the serve loop tries them. */
export function planCandidates(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  ctx: PlanContext,
): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    for (const model of probe.models) {
      const key = `${probe.provider.id}/${model.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
    void providerIndex;
  });
  const live = ctx.exhausted === undefined ? out : out.filter((c) => !ctx.exhausted!.has(c.probe.provider.id));
  if (live.length > 0) {
    out.length = 0;
    out.push(...live);
  }
  const providerOrder = new Map(probes.map((p, i) => [p.provider.id, i]));
  const jsonPenalty = (c: Candidate): number => (ctx.clientJsonMode && c.probe.provider.wire === 'anthropic' ? 1 : 0);
  const wirePenalty = (c: Candidate): number => (c.probe.provider.wire === ctx.clientWire ? 0 : 1);
  out.sort((a, b) => {
    if (b.rating !== a.rating && jsonPenalty(a) === jsonPenalty(b)) return b.rating - a.rating;
    if (jsonPenalty(a) !== jsonPenalty(b)) return jsonPenalty(a) - jsonPenalty(b);
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (wirePenalty(a) !== wirePenalty(b)) return wirePenalty(a) - wirePenalty(b);
    const calls = (ctx.callsMade.get(a.probe.provider.id) ?? 0) - (ctx.callsMade.get(b.probe.provider.id) ?? 0);
    if (calls !== 0) return calls;
    const order = (providerOrder.get(a.probe.provider.id) ?? 0) - (providerOrder.get(b.probe.provider.id) ?? 0);
    if (order !== 0) return order;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}
