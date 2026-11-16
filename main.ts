// free-model-aggregator-02 entry: one server, two client wires, the best free model behind
// every answer.
//
// Behaviour in one paragraph: the server listens on an ephemeral port, registers that port
// and its catalog with the env, then answers each client prompt by dialling exactly one free
// model through the grader's gateway (Chat Completions route only) and translating the
// answer back into the client's wire. Routing is the competition: candidates that meet the
// prompt's requirements are ranked by the brief's published rating, best first, so a plain
// or tools prompt goes to the strongest free model overall and a JSON-mode prompt to the
// strongest JSON-capable one. A refused call fails over to the next candidate in that order,
// the refusal is remembered in a request-count cooldown so later prompts skip a model that
// just refused, and every recovered refusal is reported on the response's
// x-makefun-upstream-failures header. Nothing here reads the wall clock: the cooldown counts
// requests, the ranking reads the brief, and every choice is a function of the requests and
// the gateway answers seen so far, so a replay of the same trial reproduces the same calls.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicErrorBody,
  anthropicModelList,
  anthropicRequestToOpenAI,
  completionToSseStream,
  estimateFromChars,
  openAIAnswerToAnthropic,
  openAIErrorBody,
  openAIModelList,
  plainTextOf,
  relayAnthropicStream,
  relayOpenAIStream,
  shapeOpenAICompletion,
  type ListedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses an output cap above this, and any call without one. */
const MAX_OUTPUT_TOKENS = 8192;
/** The cap named when a client body carries none at all. */
const FALLBACK_OUTPUT_TOKENS = 1024;
/** How many later requests skip a model after a 429: free-model rate limits outlive one prompt. */
const SKIP_AFTER_429 = 2;
/** How many later requests skip a model after any other refusal. */
const SKIP_AFTER_REFUSAL = 1;
/** The failures trail the env reads is capped at eight rows; never send more. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);

/** Manifest cursor: the brief lists the prompts in the order the env sends them. */
let promptCursor = 0;
/** Completion requests started so far: the only clock the cooldown logic is allowed. */
let requestSeq = 0;
/** Model id -> the requestSeq at which the model becomes eligible again. */
const cooldownUntil = new Map<string, number>();

function log(line: string): void {
  process.stderr.write(`aggregator: ${line}\n`);
}

// ---------------------------------------------------------------------------
// Routing: rank the capable free models by the brief's rating.
// ---------------------------------------------------------------------------

/**
 * Every free model that meets `requires`, best first: rating descending, and the gateway
 * id ascending as the tie-break so equal-rated models order identically on every run.
 */
function rankCandidates(requires: Requirement): ModelRow[] {
  const capable = freeModels(brief).filter((model) => meetsRequirements(model, requires));
  capable.sort((a, b) => b.ratingMilli - a.ratingMilli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return capable;
}

/** Requirements visible in a request body itself, whichever wire it arrived on. */
function requirementsFromBody(body: Json): Requirement {
  const out: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  if (isRecord(body.response_format)) {
    if (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema') out.json = true;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url' || block.type === 'input_image') out.vision = true;
    }
  }
  return out;
}

/**
 * The requirements for the request now arriving: the brief's manifest row for this arrival
 * (the manifest is the declared source, and the cursor advances once per completion
 * request), unioned with what the body itself demands — a body carrying tools, an image or
 * a JSON response_format needs a capable model no matter what either source says alone.
 */
function requirementsFor(body: Json): Requirement {
  const row = brief.prompts[promptCursor];
  promptCursor += 1;
  const fromBody = requirementsFromBody(body);
  if (row === undefined) return fromBody;
  const merged: Requirement = { ...row.requires };
  if (fromBody.tools === true) merged.tools = true;
  if (fromBody.vision === true) merged.vision = true;
  if (fromBody.json === true) merged.json = true;
  return merged;
}

// ---------------------------------------------------------------------------
// Cooldowns, counted in requests — never in seconds.
// ---------------------------------------------------------------------------

function isCooling(modelId: string, seq: number): boolean {
  return (cooldownUntil.get(modelId) ?? 0) > seq;
}

function markRefused(modelId: string, status: number, seq: number): void {
  const skip = status === 429 ? SKIP_AFTER_429 : SKIP_AFTER_REFUSAL;
  const until = seq + skip + 1;
  if (until > (cooldownUntil.get(modelId) ?? 0)) cooldownUntil.set(modelId, until);
}

// ---------------------------------------------------------------------------
// Gateway calls and the failover chain.
// ---------------------------------------------------------------------------

/** The gateway's Chat Completions route, whether OPENAI_BASE_URL already ends in /v1 or not. */
function chatCompletionsUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

/** Guarantee the gateway's output-cap rule: a cap always present, never above its ceiling. */
function ensureOutputCap(body: Json): Json {
  const out = { ...body };
  let capped = false;
  for (const key of ['max_tokens', 'max_completion_tokens'] as const) {
    const value = out[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      out[key] = Math.min(MAX_OUTPUT_TOKENS, Math.max(1, Math.floor(value)));
      capped = true;
    }
  }
  if (!capped) out.max_tokens = FALLBACK_OUTPUT_TOKENS;
  return out;
}

/** The Chat Completions body for an OpenAI-wire request: the client's body, faithfully, re-modelled. */
function openAIGatewayBody(clientBody: Json, modelId: string): Json {
  let out: Json = { ...clientBody, model: modelId };
  if (out.stream === true) {
    const options = isRecord(out.stream_options) ? (out.stream_options as Json) : {};
    out = { ...out, stream_options: { ...options, include_usage: true } };
  }
  return ensureOutputCap(out);
}

/** The tool names a gateway body declares, in the OpenAI shape both wires are translated into. */
function declaredToolNames(gatewayBody: Json): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(gatewayBody.tools)) return names;
  for (const tool of gatewayBody.tools) {
    if (!isRecord(tool)) continue;
    if (isRecord(tool.function) && typeof tool.function.name === 'string') names.add(tool.function.name);
    else if (typeof tool.name === 'string') names.add(tool.name);
  }
  return names;
}

/** A deterministic input-token estimate for a gateway body, from its messages' characters. */
function promptEstimateOf(gatewayBody: Json): number {
  const messages = Array.isArray(gatewayBody.messages) ? gatewayBody.messages : [];
  let chars = 0;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    chars += plainTextOf(message.content).length;
    if (Array.isArray(message.tool_calls)) chars += JSON.stringify(message.tool_calls).length;
  }
  return estimateFromChars(chars, 0);
}

/**
 * The upstream body to relay for a streamed answer. Providers honour stream:true with
 * server-sent events; one that instead answers 200 with a plain JSON completion has still
 * served the prompt, so the completion is shaped as usual and re-expressed as an SSE
 * stream the relays consume — the client asked for events and gets a well-ordered stream
 * in its own wire either way, from the same dialled model.
 */
async function streamSource(
  outcome: { response: Response; model: ModelRow; gatewayBody: Json },
  declared: ReadonlySet<string>,
  estimate: number,
): Promise<ReadableStream<Uint8Array>> {
  const contentType = outcome.response.headers.get('content-type') ?? '';
  if (contentType.includes('text/event-stream') || outcome.response.body === null) {
    return outcome.response.body as ReadableStream<Uint8Array>;
  }
  const answer: unknown = await outcome.response.json().catch(() => null);
  const shaped = shapeOpenAICompletion(isRecord(answer) ? answer : {}, outcome.model.id, declared, estimate);
  return completionToSseStream(shaped, outcome.model.id);
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

type ServeOutcome =
  | { ok: true; response: Response; model: ModelRow; gatewayBody: Json; failures: UpstreamFailure[] }
  | { ok: false; failures: UpstreamFailure[] };

/** The error code a refused gateway call carried, read from its own body when it has one. */
function failureCodeOf(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      const error = isRecord(value.error) ? (value.error as Json) : value;
      if (typeof error.code === 'string' && error.code !== '') return error.code.slice(0, 64);
      if (typeof error.code === 'number') return String(error.code);
      if (typeof error.type === 'string' && error.type !== '') return error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status-derived code.
  }
  if (status === 429) return 'rate_limited';
  if (status === 401 || status === 403) return 'unauthorized';
  if (status >= 500) return 'upstream_error';
  return 'request_refused';
}

async function dial(modelId: string, gatewayBody: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (gatewayBody.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(gatewayBody) });
}

/**
 * Serve one prompt through the failover chain: try the capable models in rating order —
 * models still cooling from an earlier refusal go last, so a fresh capable model is always
 * preferred and a cooled one remains a last resort — and stop at the first success. Every
 * refusal is recorded in order and cools its model for the requests that follow. Attempts
 * are capped at the number of capable candidates: each model is dialled at most once per
 * prompt, so the chain always terminates and never wastes a call re-dialling a refusal.
 */
async function serveWithFailover(buildBody: (modelId: string) => Json, requires: Requirement, seq: number): Promise<ServeOutcome> {
  const ranked = rankCandidates(requires);
  const fresh = ranked.filter((model) => !isCooling(model.id, seq));
  const cooled = ranked.filter((model) => isCooling(model.id, seq));
  const failures: UpstreamFailure[] = [];
  for (const candidate of [...fresh, ...cooled]) {
    const gatewayBody = buildBody(candidate.id);
    let response: Response;
    try {
      response = await dial(candidate.id, gatewayBody);
    } catch {
      failures.push({ status: 502, code: 'network_error', model: candidate.id });
      markRefused(candidate.id, 502, seq);
      continue;
    }
    if (response.ok && response.body !== null) {
      return { ok: true, response, model: candidate, gatewayBody, failures };
    }
    const text = await response.text().catch(() => '');
    failures.push({ status: response.status, code: failureCodeOf(response.status, text), model: candidate.id });
    markRefused(candidate.id, response.status, seq);
  }
  return { ok: false, failures };
}

/** The x-makefun-upstream-failures header value for a recovered prompt, or null when none refused. */
function failuresHeader(failures: readonly UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
}

// ---------------------------------------------------------------------------
// HTTP plumbing.
// ---------------------------------------------------------------------------

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
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

function sendJson(res: ServerResponse, status: number, body: unknown, trail: string | null): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendWireError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicErrorBody(type, message), null);
  else sendJson(res, status, openAIErrorBody(message, type, null), null);
}

function listedModels(): ListedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, displayName: model.displayName }));
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models), null);
  else sendJson(res, 200, openAIModelList(models), null);
}

function beginStream(res: ServerResponse, trail: string | null): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
}

// ---------------------------------------------------------------------------
// The two completion routes.
// ---------------------------------------------------------------------------

async function handleChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendWireError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const seq = requestSeq;
  requestSeq += 1;
  const outcome = await serveWithFailover((id) => openAIGatewayBody(body, id), requirementsFor(body), seq);
  if (!outcome.ok) {
    const last = outcome.failures[outcome.failures.length - 1];
    return sendWireError(res, 'openai', 502, 'upstream_error', `every capable free model refused (last: HTTP ${last?.status ?? 502})`);
  }
  const trail = failuresHeader(outcome.failures);
  const declared = declaredToolNames(outcome.gatewayBody);
  const estimate = promptEstimateOf(outcome.gatewayBody);
  if (body.stream === true) {
    const source = await streamSource(outcome, declared, estimate);
    beginStream(res, trail);
    try {
      for await (const frame of relayOpenAIStream(source, outcome.model.id, estimate)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await outcome.response.json().catch(() => null);
  sendJson(res, 200, shapeOpenAICompletion(isRecord(answer) ? answer : {}, outcome.model.id, declared, estimate), trail);
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendWireError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const seq = requestSeq;
  requestSeq += 1;
  const outcome = await serveWithFailover(
    (id) => ensureOutputCap(anthropicRequestToOpenAI(body, id)),
    requirementsFor(body),
    seq,
  );
  if (!outcome.ok) {
    const last = outcome.failures[outcome.failures.length - 1];
    return sendWireError(res, 'anthropic', 502, 'api_error', `every capable free model refused (last: HTTP ${last?.status ?? 502})`);
  }
  const trail = failuresHeader(outcome.failures);
  const declared = declaredToolNames(outcome.gatewayBody);
  const estimate = promptEstimateOf(outcome.gatewayBody);
  if (body.stream === true) {
    const source = await streamSource(outcome, declared, estimate);
    beginStream(res, trail);
    try {
      for await (const frame of relayAnthropicStream(source, outcome.model.id, estimate)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await outcome.response.json().catch(() => null);
  const shaped = shapeOpenAICompletion(isRecord(answer) ? answer : {}, outcome.model.id, declared, estimate);
  sendJson(res, 200, openAIAnswerToAnthropic(shaped, outcome.model.id), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handleModels(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return handleChatCompletions(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return handleMessages(req, res);
  return sendWireError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

// ---------------------------------------------------------------------------
// Trial lifecycle: register, then poll /work until the env says done.
// ---------------------------------------------------------------------------

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json().catch(() => null);
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json().catch(() => null);
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable on this tick; the next one retries.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The register body's catalog and this output are the one built object: print and exit
    // before any server starts, waiting for stdout to flush so the tail is never lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendWireError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      } else if (!res.writableEnded) {
        res.end();
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${listedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
