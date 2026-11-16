// Brief parsing, the free-model table and the catalog document.
//
// The trial brief (the file named by MAKEFUN_PROMPT_FILE) is prose wrapped around two JSON
// documents. Each document carries a `schema` string under this competition's prefix, so the
// parser ignores the prose entirely and lifts out every balanced JSON object that names one:
//
//   free-model-aggregator-02/models/v1    one row per dialable free model: the gateway id
//                                         (`openrouter/<vendor>/<model>:free`), the published
//                                         rating in milli-points and the capability flags
//                                         (tools / vision / json) plus the context window.
//   free-model-aggregator-02/prompts/v1   the six prompts in the order the env will send them:
//                                         id, wire (`openai` | `anthropic`), streaming flag,
//                                         weight and the requirements a model must meet.
//
// A list may be spelled `models` or `ratings`, the manifest `prompts` or `manifest`; the parser
// looks for the arrays themselves, so either spelling (or both lists in one document) reads fine.
//
// Nothing here touches the clock or a random source: the same brief always produces the same
// catalog bytes, which is what lets `--list` output and the register body agree byte for byte
// on a live run and on its replay.

import { readFileSync } from 'node:fs';

export const SCHEMA_PREFIX = 'free-model-aggregator-02/';
export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** OpenRouter's public endpoint: the only base_url the catalog may name (never the trial gateway). */
export const PROVIDER_BASE_URL = 'https://openrouter.ai/api/v1';
/** The secret name an entrant pushes the provider key under; named in the catalog, never valued here. */
export const PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY';
export const PROVIDER_ID = 'openrouter';

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One row of the brief's free-model table. `id` is the exact string the gateway routes. */
export interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  supportsTools: boolean | null;
  supportsVision: boolean | null;
  supportsJson: boolean | null;
  free: boolean;
}

/** The capabilities a prompt demands; a model must affirm every demanded one. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One manifest row: a prompt the env will send, in send order. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what register posts, one shared object. */
export interface Catalog {
  providers: CatalogProvider[];
}

// ---------------------------------------------------------------------------
// Scanning the brief's prose for its JSON documents.
// ---------------------------------------------------------------------------

/** Index just past the `}` that closes the object opening at `start`, or -1 when it never closes. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
      continue;
    }
    if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that parses and carries a string `schema`, in order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    let value: unknown;
    try {
      value = JSON.parse(text.slice(at, end));
    } catch {
      continue;
    }
    if (isRecord(value) && typeof value.schema === 'string') found.push(value);
  }
  return found;
}

/** The first document under this competition's prefix holding an array under any of `keys`. */
function documentWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!(doc.schema as string).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Row decoding. Tolerant of the brief's spellings, strict about the id: a row
// without a routable id is dropped rather than guessed at.
// ---------------------------------------------------------------------------

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBoolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function decodeModel(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const bare = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (bare === '') return null;
  const provider = asString(value.provider) !== '' ? asString(value.provider) : asString(value.wire);
  // A row that names the provider but spells the id bare is routed under the provider's prefix.
  const id = provider === PROVIDER_ID && !bare.toLowerCase().startsWith('openrouter/') ? `openrouter/${bare}` : bare;
  return {
    id,
    provider: provider === '' ? PROVIDER_ID : provider,
    displayName: asString(value.display_name) !== '' ? asString(value.display_name) : bare,
    ratingMilli: asFiniteNumber(value.rating_milli) ?? 0,
    contextLength: asFiniteNumber(value.context_length),
    supportsTools: asBoolOrNull(value.tools),
    supportsVision: asBoolOrNull(value.vision),
    supportsJson: asBoolOrNull(value.json),
    free: value.free !== false,
  };
}

function decodeRequirement(raw: unknown): Requirement {
  if (!isRecord(raw)) return {};
  const out: Requirement = {};
  if (raw.tools === true) out.tools = true;
  if (raw.vision === true) out.vision = true;
  if (raw.json === true) out.json = true;
  const min = asFiniteNumber(raw.min_context);
  if (min !== null) out.minContext = min;
  return out;
}

function decodePrompt(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: decodeRequirement(value.requires),
  };
}

/** Read and decode the brief at `path`. Failures land in `error` with empty lists, never a throw. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief holds no ${MODELS_SCHEMA} document` };
  const models: ModelRow[] = [];
  for (const raw of modelsDoc.doc[modelsDoc.key] as unknown[]) {
    const row = decodeModel(raw);
    if (row !== null) models.push(row);
  }
  const promptsDoc = documentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief holds no ${PROMPTS_SCHEMA} document` };
  const prompts: PromptRow[] = [];
  for (const raw of promptsDoc.doc[promptsDoc.key] as unknown[]) {
    const row = decodePrompt(raw);
    if (row !== null) prompts.push(row);
  }
  return { models, prompts, error: null };
}

// ---------------------------------------------------------------------------
// Capability matching and the catalog.
// ---------------------------------------------------------------------------

/** The brief's free models, in brief order. Paid rows are never routable. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/**
 * Whether `model` may serve a prompt with these requirements. A capability the brief leaves
 * unknown (null) never satisfies a demand: routing to an unproven model risks the whole prompt.
 */
export function meetsRequirements(model: ModelRow, requires: Requirement): boolean {
  if (requires.tools === true && model.supportsTools !== true) return false;
  if (requires.vision === true && model.supportsVision !== true) return false;
  if (requires.json === true && model.supportsJson !== true) return false;
  if (requires.minContext !== undefined && (model.contextLength === null || model.contextLength < requires.minContext)) return false;
  return true;
}

/**
 * Build the catalog from the brief: one provider, each free model once, in brief order.
 * The set is exactly the dialable ids — the catalog column compares this set (and the two
 * /v1/models answers) against the brief's fourteen, so nothing may be added or dropped here.
 */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: PROVIDER_ID,
        base_url: PROVIDER_BASE_URL,
        key_env: PROVIDER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
