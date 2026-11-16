// Wire translation between the two client APIs and the one upstream API.
//
// Clients speak OpenAI Chat Completions (POST /v1/chat/completions) or Anthropic Messages
// (POST /v1/messages), in JSON or as server-sent events. The gateway only speaks Chat
// Completions, so:
//
//   OpenAI in     -> the body passes through with the dialled model's id, and the gateway's
//                    answer (JSON or SSE) is relayed with `model` rewritten on every payload.
//   Anthropic in  -> the request is rewritten to Chat Completions (system prompt, image
//                    blocks, tool definitions, tool_use / tool_result blocks, tool choice),
//                    and the answer is rewritten back: a Messages object for JSON, or the
//                    message_start / content_block_* / message_delta / message_stop event
//                    sequence for streams, with stop reasons and usage translated.
//
// The `model` field is the scored claim, so every function here takes the id that was actually
// dialled and stamps it on every answer and every chunk — the provider spells the same model
// without the gateway's `openrouter/` prefix, and that spelling is never forwarded.
//
// Hygiene the wire column grades, applied deterministically to the provider's answer:
//   - usage is whole: a missing or fractional count is repaired, and an output count below 1
//     on an answer that carries text is raised to a character-based estimate of that text;
//   - the finish reason is one the client's API knows (an unmapped reason becomes `stop`,
//     or `tool_calls` when the answer still holds tool calls);
//   - tool calls naming a tool the client never declared are dropped.
// Every repair is a pure function of the request and the gateway's answer — no clock, no
// randomness — so a replay reproduces the same bytes. The model lists use fixed timestamps
// for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model as GET /v1/models lists it. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** Fixed `created` second for OpenAI model rows: a moving clock would break replay equality. */
const OPENAI_CREATED = 1767225600;
/** Fixed `created_at` instant for Anthropic model rows. */
const ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** Finish reasons the OpenAI wire knows. */
const OPENAI_FINISH_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

// ---------------------------------------------------------------------------
// Model lists and error envelopes.
// ---------------------------------------------------------------------------

export function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: ANTHROPIC_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

export function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

export function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Small shared helpers.
// ---------------------------------------------------------------------------

function intOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : null;
}

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A rough token estimate for a text: one token per four characters, at least the floor. */
export function estimateTokens(text: string, floor: number): number {
  return estimateFromChars(text.length, floor);
}

/** The same estimate from a character count, without materialising the text. */
export function estimateFromChars(chars: number, floor: number): number {
  return Math.max(floor, Math.ceil(chars / 4));
}

/** The concatenated text of a string body or of an array's text blocks; '' when there is none. */
export function plainTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A tool call's argument string parsed back to an object; {} when it is not a JSON object. */
function parseArguments(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: the tool input is empty.
    }
  }
  return {};
}

/** OpenAI finish_reason -> Anthropic stop_reason. */
export function anthropicStopReason(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

// ---------------------------------------------------------------------------
// Anthropic request -> Chat Completions request.
// ---------------------------------------------------------------------------

/** One Anthropic image block as an OpenAI image_url part; null for anything else. */
function imagePartOf(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' && source.media_type !== '' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result's content flattened to the string a Chat Completions tool message carries. */
function toolResultContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = plainTextOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * Translate an Anthropic Messages body into the Chat Completions body the gateway takes,
 * naming `model`. System text becomes a leading system message; tool_use blocks become
 * tool_calls on the assistant turn; each tool_result becomes its own tool message; image
 * blocks become image_url parts; tools, tool_choice, the output cap and the sampling
 * fields are carried across, and a streamed request asks the gateway for its usage chunk.
 */
export function anthropicRequestToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemText = plainTextOf(body.system) || (typeof body.system === 'string' ? body.system : '');
  if (systemText !== '') messages.push({ role: 'system', content: systemText });

  const turns = Array.isArray(body.messages) ? body.messages : [];
  for (const turn of turns) {
    if (!isRecord(turn)) continue;
    const role = turn.role === 'assistant' ? 'assistant' : 'user';
    const content = turn.content;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' && block.id !== '' ? block.id : 'toolu_unknown',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = plainTextOf(content);
      const message: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    // A user turn: tool results become tool messages first, the rest becomes content parts.
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultContent(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      const image = imagePartOf(block);
      if (image !== null) parts.push(image);
    }
    if (parts.length === 1 && parts[0].type === 'text') messages.push({ role: 'user', content: parts[0].text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chat Completions answer hygiene (JSON bodies).
// ---------------------------------------------------------------------------

/**
 * The provider's JSON answer made truthful and well-formed for the client: `model` becomes
 * the dialled id, tool calls are filtered to `declaredTools`, the finish reason is mapped
 * into the OpenAI set, and usage is made whole (see the file header). `promptEstimate`
 * stands in when the provider reported no usable input count.
 */
export function shapeOpenAICompletion(body: Json, model: string, declaredTools: ReadonlySet<string>, promptEstimate: number): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(out.choices) ? [...(out.choices as unknown[])] : [];
  let answerText = '';
  let keptToolCalls = 0;
  if (choices.length > 0 && isRecord(choices[0])) {
    const choice: Json = { ...(choices[0] as Json) };
    const message: Json = isRecord(choice.message) ? { ...(choice.message as Json) } : {};
    answerText = typeof message.content === 'string' ? message.content : plainTextOf(message.content);
    if (Array.isArray(message.tool_calls)) {
      const kept = (message.tool_calls as unknown[]).filter((call) => {
        if (!isRecord(call) || !isRecord(call.function)) return false;
        return typeof call.function.name === 'string' && declaredTools.has(call.function.name);
      });
      keptToolCalls = kept.length;
      if (kept.length > 0) message.tool_calls = kept;
      else delete message.tool_calls;
    }
    const finish = typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
    if (finish === null || !OPENAI_FINISH_REASONS.has(finish) || ((finish === 'tool_calls' || finish === 'function_call') && keptToolCalls === 0)) {
      choice.finish_reason = keptToolCalls > 0 ? 'tool_calls' : 'stop';
    }
    choice.message = message;
    choices[0] = choice;
    out.choices = choices;
  }
  const rawUsage = isRecord(out.usage) ? (out.usage as Json) : {};
  let promptTokens = intOrNull(rawUsage.prompt_tokens);
  if (promptTokens === null || promptTokens < 0) promptTokens = promptEstimate;
  let completionTokens = intOrNull(rawUsage.completion_tokens);
  if (completionTokens === null || completionTokens < 1) {
    completionTokens = answerText !== '' || keptToolCalls > 0 ? estimateTokens(answerText, 1) : 1;
  }
  out.usage = { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
  return out;
}

/** A (already shaped) Chat Completions answer as an Anthropic Messages object naming `model`. */
export function openAIAnswerToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? (choices[0] as Json) : {};
  const message = isRecord(choice.message) ? (choice.message as Json) : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : plainTextOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parseArguments(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? (body.usage as Json) : {};
  let outputTokens = intOrNull(usage.completion_tokens) ?? 0;
  if (outputTokens < 1 && text !== '') outputTokens = estimateTokens(text, 1);
  const upstreamId = typeof body.id === 'string' && body.id !== '' ? body.id.replace(/[^A-Za-z0-9_-]/g, '') : 'aggregate';
  return {
    id: `msg_${upstreamId}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: intOrNull(usage.prompt_tokens) ?? 0, output_tokens: outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events.
// ---------------------------------------------------------------------------

/** One SSE frame: an optional `event:` line, the JSON data line, the blank separator. */
export function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a decoded chunk buffer into complete frames plus the held remainder. */
function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const frames: string[] = [];
  let rest = buffer;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    frames.push(rest.slice(0, at));
    rest = rest.slice(at + 2);
    at = rest.indexOf('\n\n');
  }
  return { frames, rest };
}

/** The JSON payloads of one frame's `data:` lines. */
function framePayloads(frame: string): unknown[] {
  const out: unknown[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      out.push(JSON.parse(payload));
    } catch {
      // A non-JSON payload is relayed untouched and contributes nothing here.
    }
  }
  return out;
}

function isDoneFrame(frame: string): boolean {
  return frame.split('\n').some((line) => line.trim() === 'data: [DONE]');
}

/**
 * Relay a Chat Completions SSE stream, stamping `model` on every chunk. The terminal
 * [DONE] is held back until the stream ends so a missing usage report can be repaired:
 * when no chunk carried usage (or the reported output count is unusable on a stream that
 * carried text), one final usage chunk — empty choices, the dialect providers themselves
 * use with include_usage — is emitted first, estimated from the streamed characters.
 */
export async function* relayOpenAIStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  promptEstimate: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let heldDone: string | null = null;
  let sawUsage = false;
  let streamedChars = 0;
  let lastId = 'chatcmpl-aggregate';

  const renderFrame = (frame: string): string => {
    const lines = frame.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') return line;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        return line;
      }
      if (!isRecord(chunk)) return line;
      const next: Json = { ...chunk, model };
      if (isRecord(chunk.usage)) {
        sawUsage = true;
        const completion = intOrNull(chunk.usage.completion_tokens);
        if (completion === null || completion < 1) {
          const repaired = estimateFromChars(streamedChars, 1);
          const prompt = intOrNull(chunk.usage.prompt_tokens) ?? promptEstimate;
          next.usage = { prompt_tokens: prompt, completion_tokens: repaired, total_tokens: prompt + repaired };
        }
      }
      return `data: ${JSON.stringify(next)}`;
    });
    return `${lines.join('\n')}\n\n`;
  };

  const track = (frame: string): void => {
    for (const payload of framePayloads(frame)) {
      if (!isRecord(payload)) continue;
      if (typeof payload.id === 'string' && payload.id !== '') lastId = payload.id;
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const choice = isRecord(choices[0]) ? (choices[0] as Json) : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? (choice.delta as Json) : {};
      if (typeof delta.content === 'string') streamedChars += delta.content.length;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') {
          streamedChars += call.function.arguments.length;
        }
      }
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, rest } = splitFrames(buffer);
      buffer = rest;
      for (const frame of frames) {
        if (isDoneFrame(frame)) {
          heldDone = frame;
          continue;
        }
        track(frame);
        yield renderFrame(frame);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '' && !isDoneFrame(buffer)) {
      track(buffer);
      yield renderFrame(buffer);
    } else if (buffer.trim() !== '' && heldDone === null) {
      heldDone = buffer;
    }
  } finally {
    reader.releaseLock();
  }
  if (!sawUsage) {
    const completion = estimateFromChars(streamedChars, 1);
    yield sseFrame(null, {
      id: lastId,
      object: 'chat.completion.chunk',
      created: OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: promptEstimate, completion_tokens: completion, total_tokens: promptEstimate + completion },
    });
  }
  yield heldDone !== null ? `${heldDone.replace(/\s+$/, '')}\n\n` : 'data: [DONE]\n\n';
}

/**
 * A whole JSON completion re-expressed as a Chat Completions SSE stream: one content
 * chunk (text and tool calls together), one finish chunk, one usage chunk, [DONE].
 * Providers occasionally answer a streaming request with a plain JSON body; the handlers
 * feed that body through here so the client still receives a well-ordered stream in its
 * own wire, and both stream relays above consume it unchanged.
 */
export function completionToSseStream(body: Json, model: string): ReadableStream<Uint8Array> {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? (choices[0] as Json) : {};
  const message = isRecord(choice.message) ? (choice.message as Json) : {};
  const delta: Json = { role: 'assistant' };
  if (typeof message.content === 'string' && message.content !== '') delta.content = message.content;
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    delta.tool_calls = (message.tool_calls as unknown[]).map((call, index) => {
      const record = isRecord(call) ? call : {};
      const fn = isRecord(record.function) ? (record.function as Json) : {};
      return {
        index,
        id: typeof record.id === 'string' ? record.id : `toolu_${index}`,
        type: 'function',
        function: {
          name: typeof fn.name === 'string' ? fn.name : '',
          arguments: typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(isRecord(fn.arguments) ? fn.arguments : {}),
        },
      };
    });
  }
  const base = {
    id: typeof body.id === 'string' && body.id !== '' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: OPENAI_CREATED,
    model,
  };
  const frames = [
    sseFrame(null, { ...base, choices: [{ index: 0, delta, finish_reason: null }] }),
    sseFrame(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' }] }),
    sseFrame(null, { ...base, choices: [], usage: isRecord(body.usage) ? body.usage : { prompt_tokens: 0, completion_tokens: 1, total_tokens: 1 } }),
    'data: [DONE]\n\n',
  ];
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
}

/**
 * Translate a Chat Completions SSE stream into the Anthropic event sequence naming
 * `model`: message_start, one content block for the text (closed before any tool block),
 * one block per tool call fed by input_json_delta events, then message_delta carrying the
 * translated stop reason and the output count, then message_stop. The finish reason is
 * held until the stream's usage chunk has passed, because providers send usage last and
 * message_delta must carry the real output count; when it never arrives, the count is
 * estimated from the streamed characters so it stays a whole number of at least 1 on an
 * answer that carried text.
 */
export async function* relayAnthropicStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  promptEstimate: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexByUpstream = new Map<number, number>();
  let outputTokens = 0;
  let streamedChars = 0;

  const startFrames = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseFrame('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: promptEstimate, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeFrames = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    const open = [...toolIndexByUpstream.values()].sort((a, b) => a - b);
    for (const index of open) out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexByUpstream.clear();
    return out;
  };
  const finishFrames = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    const output = outputTokens >= 1 ? outputTokens : streamedChars > 0 ? estimateFromChars(streamedChars, 1) : outputTokens;
    return [
      ...closeFrames(),
      sseFrame('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: output },
      }),
      sseFrame('message_stop', { type: 'message_stop' }),
    ];
  };

  const framesOf = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startFrames(), ...finishFrames(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) outputTokens = numOrZero(chunk.usage.completion_tokens);
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? (choices[0] as Json) : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? (choice.delta as Json) : {};
      out.push(...startFrames());
      if (typeof delta.content === 'string' && delta.content !== '') {
        streamedChars += delta.content.length;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(
            sseFrame('content_block_start', {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            }),
          );
        }
        out.push(
          sseFrame('content_block_delta', {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: delta.content },
          }),
        );
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexByUpstream.size;
        let index = toolIndexByUpstream.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexByUpstream.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? (call.function as Json) : {};
          out.push(
            sseFrame('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? (call.function as Json) : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          streamedChars += fn.arguments.length;
          out.push(
            sseFrame('content_block_delta', {
              type: 'content_block_delta',
              index,
              delta: { type: 'input_json_delta', partial_json: fn.arguments },
            }),
          );
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = anthropicStopReason(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, rest } = splitFrames(buffer);
      buffer = rest;
      for (const frame of frames) {
        for (const event of framesOf(frame)) yield event;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of framesOf(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startFrames()) yield event;
    for (const event of finishFrames(pendingStop ?? 'end_turn')) yield event;
  }
}
