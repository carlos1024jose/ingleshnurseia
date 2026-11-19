import process from "node:process";
import {
  ALLOWED_TTS_VOICES,
  classifyGeminiError,
  ensurePlayableWav,
  getGeminiClient,
  readJsonBody,
  sendJson,
  type VercelReq,
  type VercelRes,
} from "./_gemini.js";

export default async function handler(req: VercelReq, res: VercelRes) {
  if (req.method !== "POST") {
    sendJson(res, 405, { error: "METHOD_NOT_ALLOWED", message: "Use POST." });
    return;
  }

  const ai = getGeminiClient();
  if (!ai) {
    sendJson(res, 503, {
      error: "TTS_NOT_CONFIGURED",
      message: "El servicio de voz HD no está configurado en este entorno.",
    });
    return;
  }

  const body = await readJsonBody<{
    text?: unknown;
    voiceName?: unknown;
    slow?: unknown;
  }>(req);

  const rawText = typeof body?.text === "string" ? body.text.trim() : "";
  if (!rawText) {
    sendJson(res, 400, { error: "EMPTY_TEXT", message: "Texto vacío." });
    return;
  }
  if (rawText.length > 400) {
    sendJson(res, 400, {
      error: "TEXT_TOO_LONG",
      message: "El texto para pronunciar excede el máximo de 400 caracteres.",
    });
    return;
  }

  const voiceName =
    typeof body?.voiceName === "string" && ALLOWED_TTS_VOICES.has(body.voiceName)
      ? body.voiceName
      : "Kore";
  const slow = Boolean(body?.slow);

  const configuredTtsModel = process.env.GEMINI_TTS_MODEL?.trim();
  const ttsCandidates = Array.from(
    new Set(
      [
        configuredTtsModel,
        "gemini-3.8-flash-lite-tts",
        "gemini-3.8-flash-tts",
      ].filter((m): m is string => Boolean(m))
    )
  );

  const stylePrompt = slow
    ? "Clear, warm, very slow and articulate English pronunciation instructor for nursing students. Pronounce every syllable distinctly."
    : "Clear, calm, articulate English nursing instructor speaking at a steady, easy-to-understand pace.";

  let lastError: unknown;
  for (const model of ttsCandidates) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: [
          {
            role: "user",
            parts: [
              {
                text: rawText,
                speechMetadata: { style: stylePrompt },
              },
            ],
          },
        ],
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName },
            },
          },
        },
      });

      const inlineData =
        response.candidates?.[0]?.content?.parts?.[0]?.inlineData;
      const base64Audio = inlineData?.data;

      if (!base64Audio) {
        sendJson(res, 502, {
          error: "EMPTY_AUDIO",
          message: "No se pudo generar el audio.",
        });
        return;
      }

      const playable = ensurePlayableWav(base64Audio, inlineData?.mimeType);
      sendJson(res, 200, {
        audioBase64: playable.audioBase64,
        mimeType: playable.mimeType,
        voiceName,
      });
      return;
    } catch (error) {
      lastError = error;
      const info = classifyGeminiError(error);
      if (!info.isModelNotFound) {
        break;
      }
    }
  }

  console.error("Vercel TTS error:", lastError);
  const info = classifyGeminiError(lastError);
  if (info.isAuthError) {
    sendJson(res, 401, {
      error: "TTS_AUTH_ERROR",
      message: "La clave de Gemini no es válida o no está autorizada para TTS.",
    });
    return;
  }
  if (info.isRateLimit) {
    sendJson(res, 429, {
      error: "TTS_RATE_LIMIT",
      message: "Límite temporal alcanzado en el servicio de voz HD.",
    });
    return;
  }
  sendJson(res, 502, {
    error: "TTS_UPSTREAM_ERROR",
    message: "No se pudo obtener el audio HD en este momento.",
  });
}
