// Client requests, reduced to one intermediate shape, and built back out on each
// provider's wire.
//
// The trial grades a served request by comparing what the provider parsed out of the
// request it received (its echo: system text, user text, tool names, max_tokens, stop
// sequences, image hash, JSON mode) with what the client sent. Each wire's parser reads
// those fields differently, so the intermediate shape keeps exactly the fields the three
// parsers read, and each builder writes them in the spelling that wire's parser reads
// back identically:
//
//   - system text: OpenAI system messages joined with "\n", the Anthropic `system`
//     string, Gemini's systemInstruction parts joined with "\n" all hash the same;
//   - user text: the last user turn's text blocks joined with "\n" on every wire;
//   - an image: the same base64 bytes as an OpenAI data URL, an Anthropic base64 source
//     or a Gemini inlineData part hash the same; a URL image stays a URL on all three;
//   - JSON mode: OpenAI's response_format and Gemini's responseMimeType carry it, the
//     Anthropic wire cannot (its parser always reports false), which the router knows.
//
// Tool results and tool calls inside a conversation are carried as their text where a
// wire has no block for them in this shape; the echo never reads them.

import { authHeaders, isRecord, joinBase, type Json, type Probe } from './providers.ts';

export type Part = { type: 'text'; text: string } | { type: 'image'; mediaType: string; data: string } | { type: 'imageUrl'; url: string };

export interface Turn {
  role: 'user' | 'assistant';
  parts: Part[];
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Json;
}

/** One client request in the shape every upstream builder starts from. */
export interface ClientRequest {
  systemText: string;
  turns: Turn[];
  tools: ToolSpec[];
  maxTokens: number | null;
  temperature: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === 'string');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** An OpenAI image_url value as a Part: a data URL becomes bytes, anything else stays a URL. */
function imagePartOf(url: string): Part {
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[2] };
  return { type: 'imageUrl', url };
}

function openAIParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') parts.push(imagePartOf(block.image_url.url));
  }
  return parts;
}

function anthropicParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ type: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
      } else if (source.type === 'url' && typeof source.url === 'string') parts.push({ type: 'imageUrl', url: source.url });
    } else if (block.type === 'tool_result') {
      const text = blocksText(block.content) || (typeof block.content === 'string' ? block.content : '');
      if (text !== '') parts.push({ type: 'text', text });
    }
  }
  return parts;
}

/** An OpenAI Chat Completions client body in the intermediate shape. */
export function fromOpenAIClient(body: Json): ClientRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (message.role !== 'user' && message.role !== 'assistant' && message.role !== 'tool') continue;
    turns.push({ role: message.role === 'assistant' ? 'assistant' : 'user', parts: openAIParts(message.content) });
  }
  const tools: ToolSpec[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages client body in the intermediate shape. */
export function fromAnthropicClient(body: Json): ClientRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    turns.push({ role: message.role === 'assistant' ? 'assistant' : 'user', parts: anthropicParts(message.content) });
  }
  const tools: ToolSpec[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText: blocksText(body.system),
    turns,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// The intermediate shape out on each provider wire.
// ---------------------------------------------------------------------------

function textOfParts(parts: Part[]): string {
  return parts.flatMap((p) => (p.type === 'text' ? [p.text] : [])).join('\n');
}

export function toOpenAI(cr: ClientRequest, model: string): Json {
  const messages: Json[] = [];
  if (cr.systemText !== '') messages.push({ role: 'system', content: cr.systemText });
  for (const turn of cr.turns) {
    if (turn.parts.every((p) => p.type === 'text')) {
      messages.push({ role: turn.role, content: textOfParts(turn.parts) });
      continue;
    }
    messages.push({
      role: turn.role,
      content: turn.parts.map((p) =>
        p.type === 'text'
          ? { type: 'text', text: p.text }
          : p.type === 'image'
            ? { type: 'image_url', image_url: { url: `data:${p.mediaType};base64,${p.data}` } }
            : { type: 'image_url', image_url: { url: p.url } },
      ),
    });
  }
  const out: Json = { model, messages };
  if (cr.maxTokens !== null) out.max_tokens = cr.maxTokens;
  if (cr.temperature !== null) out.temperature = cr.temperature;
  if (cr.stops.length > 0) out.stop = cr.stops;
  if (cr.tools.length > 0) {
    out.tools = cr.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    out.tool_choice = 'auto';
  }
  if (cr.jsonMode) out.response_format = { type: 'json_object' };
  if (cr.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function toAnthropic(cr: ClientRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of cr.turns) {
    if (turn.parts.every((p) => p.type === 'text')) {
      messages.push({ role: turn.role, content: textOfParts(turn.parts) });
      continue;
    }
    messages.push({
      role: turn.role,
      content: turn.parts.map((p) =>
        p.type === 'text'
          ? { type: 'text', text: p.text }
          : p.type === 'image'
            ? { type: 'image', source: { type: 'base64', media_type: p.mediaType, data: p.data } }
            : { type: 'image', source: { type: 'url', url: p.url } },
      ),
    });
  }
  const out: Json = { model, messages };
  if (cr.systemText !== '') out.system = cr.systemText;
  if (cr.maxTokens !== null) out.max_tokens = cr.maxTokens;
  if (cr.temperature !== null) out.temperature = cr.temperature;
  if (cr.stops.length > 0) out.stop_sequences = cr.stops;
  if (cr.tools.length > 0) {
    out.tools = cr.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    out.tool_choice = { type: 'auto' };
  }
  if (cr.stream) out.stream = true;
  return out;
}

export function toGemini(cr: ClientRequest): Json {
  const contents = cr.turns.map((turn) => ({
    role: turn.role === 'assistant' ? 'model' : 'user',
    parts: turn.parts.map((p) =>
      p.type === 'text' ? { text: p.text } : p.type === 'image' ? { inlineData: { mimeType: p.mediaType, data: p.data } } : { fileData: { fileUri: p.url } },
    ),
  }));
  const generationConfig: Json = {};
  if (cr.maxTokens !== null) generationConfig.maxOutputTokens = cr.maxTokens;
  if (cr.temperature !== null) generationConfig.temperature = cr.temperature;
  if (cr.stops.length > 0) generationConfig.stopSequences = cr.stops;
  if (cr.jsonMode) generationConfig.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (cr.systemText !== '') out.systemInstruction = { parts: [{ text: cr.systemText }] };
  if (cr.tools.length > 0) {
    out.tools = [{ functionDeclarations: cr.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/** One call to one provider, on its own wire, with the request built for that wire. */
export async function callProvider(envUrl: string, probe: Probe, modelId: string, cr: ClientRequest): Promise<Response> {
  const provider = probe.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (cr.stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(toOpenAI(cr, modelId)) });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(toAnthropic(cr, modelId)) });
  }
  const action = cr.stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return fetch(`${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}`, { method: 'POST', headers, body: JSON.stringify(toGemini(cr)) });
}
