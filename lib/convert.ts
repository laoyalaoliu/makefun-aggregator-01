// Normalized answers and cross-wire synthesis.
//
// Every provider answer, in whichever wire it arrived, is reduced to one normalized shape:
// the answer text whole, the usage counts in (input, output) tokens, and a stop family
// (0 = a normal stop, 1 = a length stop, 2 = a tool-call stop, 3 = a refusal). The client
// wires are then synthesized from that shape, so a request the client sent on one wire can
// be served by a provider on any other and still read as a well-formed answer of the
// client's own API: the text whole, the usage as the provider reported it, and the stop
// reason named in the client's wire.
//
// Streaming answers are handled the same way, one level down: a provider's event stream is
// read to its end and parsed into the normalized shape, and the client's event stream is
// synthesized from it. The simulated providers answer at once, so nothing is gained by
// relaying frames as they arrive, and a synthesized stream cannot interleave two wires'
// event orders by accident. Everything here is a pure function of its arguments: no clock,
// no randomness, so a replay that sees the same provider answers emits the same bytes.

import { isRecord, type Json } from './providers.ts';

/** A provider answer reduced to the parts every client wire needs. */
export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** 0 normal, 1 length, 2 tool calls, 3 refusal. */
  stopFamily: 0 | 1 | 2 | 3;
}

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'] as const;
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'] as const;

/** Any wire's stop reason as a family index. */
export function stopFamilyOf(reason: unknown): 0 | 1 | 2 | 3 {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

export const openAIStop = (a: NormAnswer): string => OPENAI_STOPS[a.stopFamily];
export const anthropicStop = (a: NormAnswer): string => ANTHROPIC_STOPS[a.stopFamily];

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of a string or of an array's text blocks, joined with nothing between them. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

// ---------------------------------------------------------------------------
// Provider answers, JSON form, into the normalized shape.
// ---------------------------------------------------------------------------

export function normFromOpenAIJson(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(message.content),
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

export function normFromAnthropicJson(body: Json): NormAnswer {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

export function normFromGeminiJson(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join(''),
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

// ---------------------------------------------------------------------------
// Provider answers, streamed form (the whole SSE body as text), into the shape.
// ---------------------------------------------------------------------------

/** The data payloads of an SSE body, in order, each frame's data lines joined. */
function ssePayloads(bodyText: string): string[] {
  const out: string[] = [];
  for (const frame of bodyText.replace(/\r\n/g, '\n').split('\n\n')) {
    const data = frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''));
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** The named events of an SSE body: (event name, parsed JSON) pairs in order. */
function sseEvents(bodyText: string): { event: string; data: Json }[] {
  const out: { event: string; data: Json }[] = [];
  for (const frame of bodyText.replace(/\r\n/g, '\n').split('\n\n')) {
    let event = '';
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    if (dataLines.length === 0) continue;
    try {
      const value: unknown = JSON.parse(dataLines.join('\n'));
      if (isRecord(value)) out.push({ event, data: value });
    } catch {
      // A frame that is not JSON carries no answer text.
    }
  }
  return out;
}

export function normFromOpenAIStream(bodyText: string): NormAnswer {
  let text = '';
  let input = 0;
  let output = 0;
  let family: NormAnswer['stopFamily'] = 0;
  for (const payload of ssePayloads(bodyText)) {
    if (payload.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = num(chunk.usage.prompt_tokens);
      output = num(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') family = stopFamilyOf(choice.finish_reason);
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

export function normFromAnthropicStream(bodyText: string): NormAnswer {
  let text = '';
  let input = 0;
  let output = 0;
  let family: NormAnswer['stopFamily'] = 0;
  for (const { event, data } of sseEvents(bodyText)) {
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = num(data.message.usage.input_tokens);
    } else if (event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    } else if (event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') family = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) output = num(data.usage.output_tokens);
    }
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

export function normFromGeminiStream(bodyText: string): NormAnswer {
  let text = '';
  let input = 0;
  let output = 0;
  let family: NormAnswer['stopFamily'] = 0;
  for (const payload of ssePayloads(bodyText)) {
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
      if (typeof candidate.finishReason === 'string') family = stopFamilyOf(candidate.finishReason);
    }
    if (isRecord(value.usageMetadata)) {
      input = num(value.usageMetadata.promptTokenCount);
      output = num(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

// ---------------------------------------------------------------------------
// The normalized shape out as each client wire, JSON and streamed forms.
// ---------------------------------------------------------------------------

/** A fixed epoch second for synthesized documents: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** Split a text the way the simulated providers do: even 13-character pieces. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export function openAIJsonFromNorm(a: NormAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: a.text }, finish_reason: openAIStop(a), logprobs: null }],
    usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
  };
}

export function anthropicJsonFromNorm(a: NormAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: a.text }],
    stop_reason: anthropicStop(a),
    stop_sequence: null,
    usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens },
  };
}

export function openAIStreamFromNorm(a: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sse(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces(a.text)) frames.push(sse(null, chunk({ content: piece }, null)));
  frames.push(sse(null, chunk({}, openAIStop(a))));
  frames.push(
    sse(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function anthropicStreamFromNorm(a: NormAnswer, model: string): string {
  const frames = [
    sse('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: a.inputTokens, output_tokens: 0 },
      },
    }),
    sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces(a.text)) {
    frames.push(sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(a), stop_sequence: null }, usage: { output_tokens: a.outputTokens } }),
    sse('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
