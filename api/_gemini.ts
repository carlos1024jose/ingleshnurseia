import { Buffer } from "node:buffer";
import process from "node:process";
import type { IncomingMessage } from "node:http";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

export interface VercelReq extends Partial<IncomingMessage> {
  method?: string;
  body?: unknown;
  headers?: Record<string, string | string[] | undefined>;
}

export interface VercelRes {
  statusCode?: number;
  status?: (code: number) => VercelRes;
  json?: (data: unknown) => void;
  setHeader?: (name: string, value: string) => void;
  end?: (chunk?: string) => void;
}

export interface AssistantHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export const SYSTEM_INSTRUCTION = `Eres el asistente educativo de English Nurse AI, una plataforma digital dirigida a estudiantes de Enfermería que aprenden inglés técnico del área de la salud.

Tu rol es estrictamente educativo:
1. Explicar el significado de términos médicos y de Enfermería en inglés con traducción precisa al español.
2. Aclarar diferencias entre palabras similares o que suelen confundirse (por ejemplo: pain vs ache, symptom vs sign, dosage vs prescription, IV vs IM).
3. Proporcionar ejemplos sencillos y realistas de uso en entorno clínico (incluyendo oración en inglés y traducción al español).
4. Explicar expresiones y abreviaturas comunes utilizadas por personal de enfermería y en la comunicación con pacientes.

Reglas de estilo y seguridad:
- Responde en español claro, estructurado y fácil de leer para estudiantes de Enfermería.
- Incluye los términos técnicos y ejemplos clave en inglés resaltados con claridad.
- Mantén un tono pedagógico, directo y profesional.
- NO des diagnósticos médicos, prescripciones ni recomendaciones clínicas personalizadas. Recuerda brevemente cuando corresponda que tus explicaciones tienen un propósito educativo y no sustituyen la orientación de profesionales de la salud o docentes clínicos.`;

export const ALLOWED_TTS_VOICES = new Set(["Kore", "Zephyr", "Puck", "Charon"]);

const DEFAULT_TEXT_MODELS = ["gemini-4-hydrogen", "gemini-3.1-flash-lite"];
let cachedDiscoveredModel: string | null = null;

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
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY?.trim(),
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
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
        if (n.includes("hydrogen")) return 0;
        if (n.includes("flash-lite")) return 1;
        if (n.includes("flash")) return 2;
        return 3;
      };
      return score(a) - score(b);
    });

    return discovered.slice(0, 3);
  } catch {
    return [];
  }
}

export async function generateNursingReply(
  ai: GoogleGenAI,
  question: string,
  safeHistory: AssistantHistoryItem[]
): Promise<{ reply: string; modelUsed: string }> {
  const contents = [
    ...safeHistory.map((msg) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    })),
    {
      role: "user",
      parts: [{ text: question }],
    },
  ];

  const configuredModel = process.env.GEMINI_MODEL?.trim();
  const initialCandidates = Array.from(
    new Set(
      [configuredModel, cachedDiscoveredModel, ...DEFAULT_TEXT_MODELS].filter(
        (m): m is string => Boolean(m)
      )
    )
  );

  const tryModels = async (models: string[]) => {
    let lastError: unknown;
    for (const model of models) {
      try {
        const supportsThinkingLevel =
          model === "gemini-4-hydrogen" || model.startsWith("gemini-3.");
        const response = await ai.models.generateContent({
          model,
          contents,
          config: supportsThinkingLevel
            ? {
                systemInstruction: SYSTEM_INSTRUCTION,
                thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
              }
            : {
                systemInstruction: SYSTEM_INSTRUCTION,
              },
        });
        cachedDiscoveredModel = model;
        return { reply: response.text?.trim() ?? "", modelUsed: model };
      } catch (error) {
        lastError = error;
        const info = classifyGeminiError(error);
        if (!info.isModelNotFound) {
          throw error;
        }
      }
    }
    throw lastError ?? new Error("No available Gemini model succeeded.");
  };

  try {
    return await tryModels(initialCandidates);
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

    return await tryModels(discovered);
  }
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
