// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the trial brief (the provider directory, the sealed quality table
// and the workload manifest), probes every provider's catalog in its own wire, builds
// one catalog document, registers it with the env, and then serves the client's twenty
// requests, one at a time:
//
//   GET  /v1/models            the free models it serves, in the shape of the auth
//                              header it got (x-api-key: Anthropic, else OpenAI)
//   POST /v1/chat/completions  an OpenAI-shaped client request
//   POST /v1/messages          an Anthropic-shaped client request
//
// Each request is planned by lib/router.ts (best-rated free model that meets the
// request's requirements first) and served by lib/upstream.ts on the chosen provider's
// own wire (OpenAI, Anthropic or Gemini), with the answer returned in the client's
// wire by lib/convert.ts. A provider refusal (402, 429, 5xx) walks the plan to the
// next candidate: a refusal is never the client's answer. When the provider speaks
// the client's wire, the client's body passes through with only the model name
// changed (plus stream_options.include_usage on a streamed OpenAI call, so the
// stream carries the usage the client checks), and the provider's answer passes
// back untouched.
//
// `node main.ts --list` prints the catalog document and exits 0; the register body
// carries that same object, byte for byte. Everything is a function of the brief,
// the requests and the providers' answers: no wall clock, no unseeded randomness,
// so a replayed trial walks the same plan and leaves the same state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import { anthropicModelList, anthropicError, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  anthropicJsonFromNorm,
  anthropicStreamFromNorm,
  normFromAnthropicJson,
  normFromAnthropicStream,
  normFromGeminiJson,
  normFromGeminiStream,
  normFromOpenAIJson,
  normFromOpenAIStream,
  openAIJsonFromNorm,
  openAIStreamFromNorm,
  type NormAnswer,
} from './lib/convert.ts';
import { callProvider, fromAnthropicClient, fromOpenAIClient, type ClientRequest } from './lib/upstream.ts';
import { planCandidates, type Candidate } from './lib/router.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Client API requests served so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Provider calls made so far, by provider id: the router spreads calls under budgets. */
const callsMade = new Map<string, number>();
/** Providers whose own refusal said the budget is spent or that they are out for the trial. */
const exhaustedProviders = new Set<string>();

/** Learn from a refusal's own words: a spent budget and an outage never heal mid-trial. */
function noteRefusal(providerId: string, bodyText: string): void {
  if (bodyText.includes('budget is spent') || / is out\b/.test(bodyText)) exhaustedProviders.add(providerId);
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, deduplicated by id. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, cr: ClientRequest): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return {
    tools: cr.tools.length > 0,
    vision: cr.turns.some((t) => t.parts.some((p) => p.type !== 'text')),
    json: cr.jsonMode,
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Whether a provider's answer can serve: a 402, 429 or 5xx is a refusal, not an answer. */
function refused(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

function countCall(providerId: string): void {
  callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
}

/**
 * Serve one request on the same wire it arrived on: the client's body with only the
 * model changed, the provider's answer passed back untouched. Returns false when the
 * provider refused, so the caller walks the plan.
 */
async function servePassThrough(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  res: ServerResponse,
): Promise<boolean> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  const upstreamBody: Json = { ...body, model: candidate.model.id };
  let url: string;
  if (clientWire === 'openai') {
    if (stream) upstreamBody.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
    url = `${base}/v1/chat/completions`;
  } else {
    headers['anthropic-version'] = '2023-06-01';
    url = `${base}/v1/messages`;
  }
  countCall(provider.id);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  } catch {
    return false;
  }
  if (!upstream.ok || refused(upstream.status)) {
    noteRefusal(provider.id, await upstream.text().catch(() => ''));
    return false;
  }
  const contentType = upstream.headers.get('content-type') ?? (stream ? 'text/event-stream' : 'application/json');
  res.writeHead(200, { 'content-type': contentType, ...(stream ? { 'cache-control': 'no-cache', connection: 'keep-alive' } : {}) });
  if (upstream.body !== null && stream) {
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
    }
    res.end();
    return true;
  }
  res.end(await upstream.text());
  return true;
}

/**
 * Serve one request across wires: build the request on the provider's wire, reduce
 * its answer to the normalized shape, and synthesize the client's wire from it.
 * Returns false when the provider refused.
 */
async function serveCrossWire(candidate: Candidate, clientWire: 'openai' | 'anthropic', cr: ClientRequest, res: ServerResponse): Promise<boolean> {
  const provider = candidate.probe.provider;
  countCall(provider.id);
  let upstream: Response;
  try {
    upstream = await callProvider(ENV_URL, candidate.probe, candidate.model.id, cr);
  } catch {
    return false;
  }
  if (!upstream.ok || refused(upstream.status)) {
    noteRefusal(provider.id, await upstream.text().catch(() => ''));
    return false;
  }
  const wire = provider.wire;
  let norm: NormAnswer;
  if (cr.stream) {
    const text = await upstream.text();
    norm = wire === 'openai' ? normFromOpenAIStream(text) : wire === 'anthropic' ? normFromAnthropicStream(text) : normFromGeminiStream(text);
  } else {
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) return false;
    norm = wire === 'openai' ? normFromOpenAIJson(answer) : wire === 'anthropic' ? normFromAnthropicJson(answer) : normFromGeminiJson(answer);
  }
  if (norm.text === '') return false;
  if (cr.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(clientWire === 'openai' ? openAIStreamFromNorm(norm, candidate.model.id) : anthropicStreamFromNorm(norm, candidate.model.id));
    return true;
  }
  sendJson(res, 200, clientWire === 'openai' ? openAIJsonFromNorm(norm, candidate.model.id) : anthropicJsonFromNorm(norm, candidate.model.id));
  return true;
}

/** Walk the plan for one request until a provider serves it. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const cr = clientWire === 'openai' ? fromOpenAIClient(body) : fromAnthropicClient(body);
  const requires = requirementsFor(body, cr);
  const plan = planCandidates(probes, brief.ratings, requires, { clientWire, clientJsonMode: cr.jsonMode, callsMade, exhausted: exhaustedProviders });
  if (plan.length === 0) {
    sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  for (const candidate of plan) {
    const sameWire = candidate.probe.provider.wire === clientWire;
    const served = sameWire
      ? await servePassThrough(candidate, clientWire, body, cr.stream, res)
      : await serveCrossWire(candidate, clientWire, cr, res);
    if (served) return;
    if (res.headersSent) return; // A stream that started cannot be re-served; nothing in the plan reaches here.
  }
  sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free provider that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) {
      sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    await serve(wire, body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
