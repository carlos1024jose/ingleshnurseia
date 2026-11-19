import { Buffer } from "node:buffer";
import process from "node:process";
import type { IncomingMessage } from "node:http";
import { performance } from "node:perf_hooks";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

export interface VercelReq extends Partial<IncomingMessage> {
  method?: string;
  body?: unknown;
  headers?: Record<string, string | string[] | undefined>;
}

export interface VercelRes {
  statusCode?: number;
  writableEnded?: boolean;
  status?: (code: number) => VercelRes;
  json?: (data: unknown) => void;
  setHeader?: (name: string, value: string) => unknown;
  flushHeaders?: () => void;
  write?: (chunk: string) => boolean;
  end?: (chunk?: string) => void;
  on?: (event: string, listener: () => void) => unknown;
}

export interface AssistantHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export type QueryComplexity = "simple" | "complex";

export interface GenerationMetrics {
  tier: QueryComplexity;
  modelUsed: string;
  backendPrepMs: number;
  promptPrepMs: number;
  sdkDispatchMs: number;
  ttftMs: number;
  genDurationMs: number;
  totalMs: number;
  promptTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  inputChars: number;
  outputChars: number;
  historyTurnsUsed: number;
}

// Concise, medically accurate prompts tuned by query tier to minimize input tokens & latency
const SYSTEM_INSTRUCTION_SIMPLE = `Eres el asistente educativo de English Nurse AI para estudiantes de Enfermería.
Responde en español claro y directo, resaltando términos y ejemplos clave en inglés.
Para definiciones, vocabulario o siglas: da traducción precisa, uso clínico breve y 1 ejemplo corto inglés-español (máx. ~120 palabras).
Solo propósito educativo: no des diagnósticos ni prescripciones médicas.`;

const SYSTEM_INSTRUCTION_COMPLEX = `Eres el asistente educativo de English Nurse AI para estudiantes de Enfermería.
Responde en español claro y estructurado, destacando términos y frases clave en inglés.
Explica con precisión diferencias entre términos, procedimientos o comunicación clínica, con ejemplos prácticos inglés-español (máx. ~280 palabras salvo que se pida mayor detalle).
Solo propósito educativo: no emitas diagnósticos ni indicaciones clínicas personalizadas y recuerda brevemente que es apoyo académico.`;

export const ALLOWED_TTS_VOICES = new Set(["Kore", "Zephyr", "Puck", "Charon"]);

const SIMPLE_TIER_MODELS = [
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
  "gemini-4-hydrogen",
];

const COMPLEX_TIER_MODELS = [
  "gemini-3.8-flash",
  "gemini-4-hydrogen",
  "gemini-3.1-flash-lite",
];

let cachedClient: GoogleGenAI | null = null;
let cachedApiKey: string | null = null;
const cachedModelByTier: Partial<Record<QueryComplexity, string>> = {};

export function sendJson(
  res: VercelRes,
  statusCode: number,
  payload: unknown
): void {
  if (typeof res.status === "function") {
    const chained = res.status(statusCode);
    if (chained && typeof chained.json === "function") {
      chained.json(payload);
      return;
    }
  }
  res.statusCode = statusCode;
  res.setHeader?.("Content-Type", "application/json; charset=utf-8");
  res.end?.(JSON.stringify(payload));
}

export function isGeminiConfigured(): boolean {
  const key = process.env.GEMINI_API_KEY?.trim();
  return Boolean(
    key &&
      key !== "MY_GEMINI_API_KEY" &&
      key !== "YOUR_API_KEY" &&
      key.length > 10
  );
}

export function getGeminiClient(): GoogleGenAI | null {
  if (!isGeminiConfigured()) {
    return null;
  }
  const key = process.env.GEMINI_API_KEY!.trim();
  if (cachedClient && cachedApiKey === key) {
    return cachedClient;
  }
  cachedApiKey = key;
  cachedClient = new GoogleGenAI({
    apiKey: key,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
  return cachedClient;
}

export async function readJsonBody<T = Record<string, unknown>>(
  req: VercelReq
): Promise<T> {
  const raw = req.body;
  if (raw !== undefined && raw !== null) {
    if (typeof raw === "string") {
      try {
        return JSON.parse(raw) as T;
      } catch {
        return {} as T;
      }
    }
    if (Buffer.isBuffer(raw)) {
      try {
        return JSON.parse(raw.toString("utf8")) as T;
      } catch {
        return {} as T;
      }
    }
    if (typeof raw === "object") {
      return raw as T;
    }
  }

  if (typeof req.on === "function") {
    try {
      const chunks: Buffer[] = [];
      for await (const chunk of req as AsyncIterable<Buffer | string>) {
        chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
      }
      const text = Buffer.concat(chunks).toString("utf8").trim();
      if (text) {
        return JSON.parse(text) as T;
      }
    } catch {
      return {} as T;
    }
  }

  return {} as T;
}

/**
 * Zero-latency deterministic heuristic classifier (no extra LLM call).
 */
export function classifyNursingQuery(question: string): QueryComplexity {
  const normalized = question.trim().toLowerCase();
  const wordCount = normalized.split(/\s+/).filter(Boolean).length;

  const complexPatterns =
    /\b(diferencia|diferencias|compara|comparar|versus|vs\.?|procedimiento|protocolo|paso a paso|pasos|explica|explicame|explícame|profundidad|detallad|cuando usar|cuándo usar|ejemplos|diálogo|dialogo|conversación|conversacion|valoración|valoracion|administración|administracion|cuidados|caso clínico|caso clinico|diagnóstico|diagnostico|dame \d+)\b/i;

  if (normalized.length >= 130 || wordCount >= 22 || complexPatterns.test(normalized)) {
    return "complex";
  }
  return "simple";
}

function buildOptimizedPromptPayload(
  question: string,
  safeHistory: AssistantHistoryItem[],
  tier: QueryComplexity
) {
  const referencesPriorContext =
    /\b(eso|esa|ese|esos|esas|anterior|antes|más ejemplos|mas ejemplos|otra vez|también|tambien|y qué|y que)\b/i.test(
      question
    );

  const maxTurns = tier === "simple" ? (referencesPriorContext ? 2 : 0) : 4;
  const trimmedHistory = maxTurns > 0 ? safeHistory.slice(-maxTurns) : [];

  const contents = [
    ...trimmedHistory.map((msg) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [
        {
          text:
            msg.role === "assistant" && msg.content.length > 500
              ? `${msg.content.slice(0, 500)}...`
              : msg.content,
        },
      ],
    })),
    {
      role: "user",
      parts: [{ text: question }],
    },
  ];

  const systemInstruction =
    tier === "simple" ? SYSTEM_INSTRUCTION_SIMPLE : SYSTEM_INSTRUCTION_COMPLEX;

  return {
    contents,
    systemInstruction,
    historyTurnsUsed: trimmedHistory.length,
  };
}

function getModelConfig(
  model: string,
  tier: QueryComplexity,
  systemInstruction: string
) {
  const lower = model.toLowerCase();
  if (lower.includes("flash-lite")) {
    return {
      systemInstruction,
      thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
    };
  }
  if (lower === "gemini-4-hydrogen" || lower.startsWith("gemini-3.")) {
    return {
      systemInstruction,
      thinkingConfig: {
        thinkingLevel:
          tier === "simple" ? ThinkingLevel.LOW : ThinkingLevel.LOW,
      },
    };
  }
  return { systemInstruction };
}

export function classifyGeminiError(error: unknown): {
  status?: number;
  message: string;
  isAuthError: boolean;
  isRateLimit: boolean;
  isModelNotFound: boolean;
} {
  const err = error as { status?: number; code?: number; message?: string };
  const status = err?.status ?? err?.code;
  const message = err?.message ?? "Error desconocido";
  const lower = message.toLowerCase();

  const isAuthError =
    status === 401 ||
    status === 403 ||
    lower.includes("401") ||
    lower.includes("403") ||
    lower.includes("permission_denied") ||
    lower.includes("api_key_invalid") ||
    lower.includes("api key not valid") ||
    lower.includes("unauthenticated") ||
    lower.includes("unauthorized");

  const isRateLimit =
    status === 429 ||
    lower.includes("429") ||
    lower.includes("resource_exhausted") ||
    lower.includes("quota");

  const isModelNotFound =
    !isAuthError &&
    !isRateLimit &&
    (status === 404 ||
      lower.includes("not_found") ||
      lower.includes("is not found") ||
      lower.includes("not supported for generatecontent") ||
      lower.includes("unknown model"));

  return { status, message, isAuthError, isRateLimit, isModelNotFound };
}

async function discoverActiveTextModels(ai: GoogleGenAI): Promise<string[]> {
  try {
    const pager = await ai.models.list();
    const discovered: string[] = [];

    for await (const m of pager) {
      const rawName = m.name ?? "";
      const id = rawName.replace(/^models\//, "").trim();
      if (!id) continue;

      const lower = id.toLowerCase();
      if (
        !lower.includes("gemini") ||
        lower.includes("tts") ||
        lower.includes("image") ||
        lower.includes("embedding") ||
        lower.includes("live") ||
        lower.includes("transcribe") ||
        lower.includes("omni") ||
        lower.includes("decision") ||
        lower.includes("robotics") ||
        lower.includes("computer-use")
      ) {
        continue;
      }

      const actions =
        (m as { supportedActions?: string[] }).supportedActions ?? [];
      if (actions.length > 0 && !actions.includes("generateContent")) {
        continue;
      }

      discovered.push(id);
    }

    discovered.sort((a, b) => {
      const score = (name: string) => {
        const n = name.toLowerCase();
        if (n.includes("3.5-flash-lite") || n.includes("3.1-flash-lite")) return 0;
        if (n.includes("flash-lite")) return 1;
        if (n.includes("3.8-flash") || n.includes("hydrogen")) return 2;
        if (n.includes("flash")) return 3;
        return 4;
      };
      return score(a) - score(b);
    });

    return discovered.slice(0, 3);
  } catch {
    return [];
  }
}

function getCandidateList(tier: QueryComplexity): string[] {
  const configuredModel = process.env.GEMINI_MODEL?.trim();
  const baseModels =
    tier === "simple" ? SIMPLE_TIER_MODELS : COMPLEX_TIER_MODELS;

  return Array.from(
    new Set(
      [configuredModel, cachedModelByTier[tier], ...baseModels].filter(
        (m): m is string => Boolean(m)
      )
    )
  );
}

export function roundMs(ms: number): number {
  return Math.round(ms * 100) / 100;
}

export async function streamNursingReply(options: {
  ai: GoogleGenAI;
  question: string;
  safeHistory: AssistantHistoryItem[];
  tReqStart: number;
  tPrepDone: number;
  signal?: AbortSignal;
  onMeta?: (meta: { tier: QueryComplexity; model: string }) => void;
  onChunk: (text: string) => void;
}): Promise<{ reply: string; metrics: GenerationMetrics }> {
  const { ai, question, safeHistory, tReqStart, tPrepDone, signal, onMeta, onChunk } =
    options;

  const tier = classifyNursingQuery(question);
  const { contents, systemInstruction, historyTurnsUsed } =
    buildOptimizedPromptPayload(question, safeHistory, tier);
  const tPromptDone = performance.now();

  const initialCandidates = getCandidateList(tier);

  const tryStreamModels = async (models: string[]) => {
    let lastError: unknown;
    for (const model of models) {
      if (signal?.aborted) {
        throw new Error("CLIENT_ABORTED");
      }

      const tDispatch = performance.now();
      let tFirstChunk: number | null = null;
      let fullReply = "";
      let promptTokens: number | undefined;
      let outputTokens: number | undefined;
      let totalTokens: number | undefined;

      try {
        const stream = await ai.models.generateContentStream({
          model,
          contents,
          config: getModelConfig(model, tier, systemInstruction),
        });

        onMeta?.({ tier, model });

        for await (const chunk of stream) {
          if (signal?.aborted) {
            break;
          }

          if (chunk.usageMetadata) {
            promptTokens = chunk.usageMetadata.promptTokenCount;
            outputTokens = chunk.usageMetadata.candidatesTokenCount;
            totalTokens = chunk.usageMetadata.totalTokenCount;
          }

          const delta = chunk.text;
          if (delta) {
            if (tFirstChunk === null) {
              tFirstChunk = performance.now();
            }
            fullReply += delta;
            onChunk(delta);
          }
        }

        const tGenDone = performance.now();
        cachedModelByTier[tier] = model;

        const metrics: GenerationMetrics = {
          tier,
          modelUsed: model,
          backendPrepMs: roundMs(tPrepDone - tReqStart),
          promptPrepMs: roundMs(tPromptDone - tPrepDone),
          sdkDispatchMs: roundMs(tDispatch - tPromptDone),
          ttftMs: roundMs((tFirstChunk ?? tGenDone) - tDispatch),
          genDurationMs: roundMs(tGenDone - tDispatch),
          totalMs: roundMs(tGenDone - tReqStart),
          promptTokens,
          outputTokens,
          totalTokens,
          inputChars: question.length,
          outputChars: fullReply.length,
          historyTurnsUsed,
        };

        return { reply: fullReply.trim(), metrics };
      } catch (error) {
        lastError = error;
        // If we already emitted partial chunks to the client, do not retry another model mid-stream
        if (tFirstChunk !== null) {
          throw error;
        }
        const info = classifyGeminiError(error);
        if (!info.isModelNotFound) {
          throw error;
        }
      }
    }

    throw lastError ?? new Error("No available Gemini model succeeded.");
  };

  try {
    return await tryStreamModels(initialCandidates);
  } catch (error) {
    const info = classifyGeminiError(error);
    if (!info.isModelNotFound) {
      throw error;
    }

    const discovered = (await discoverActiveTextModels(ai)).filter(
      (m) => !initialCandidates.includes(m)
    );
    if (discovered.length === 0) {
      throw error;
    }

    return await tryStreamModels(discovered);
  }
}

export async function generateNursingReply(
  ai: GoogleGenAI,
  question: string,
  safeHistory: AssistantHistoryItem[],
  tReqStart = performance.now(),
  tPrepDone = performance.now()
): Promise<{ reply: string; modelUsed: string; metrics: GenerationMetrics }> {
  let replyText = "";
  const { reply, metrics } = await streamNursingReply({
    ai,
    question,
    safeHistory,
    tReqStart,
    tPrepDone,
    onChunk: (delta) => {
      replyText += delta;
    },
  });
  return {
    reply: reply || replyText.trim(),
    modelUsed: metrics.modelUsed,
    metrics,
  };
}

export function ensurePlayableWav(
  base64Data: string,
  mimeType?: string
): { audioBase64: string; mimeType: string } {
  const lower = (mimeType ?? "").toLowerCase();
  if (!lower.includes("pcm") && !lower.includes("l16")) {
    return {
      audioBase64: base64Data,
      mimeType: mimeType || "audio/wav",
    };
  }

  const rateMatch = lower.match(/rate=(\d+)/);
  const sampleRate = rateMatch ? Number(rateMatch[1]) : 24000;
  const pcmBuffer = Buffer.from(base64Data, "base64");
  const header = Buffer.alloc(44);
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;

  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcmBuffer.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcmBuffer.length, 40);

  const wavBuffer = Buffer.concat([header, pcmBuffer]);
  return {
    audioBase64: wavBuffer.toString("base64"),
    mimeType: "audio/wav",
  };
}
