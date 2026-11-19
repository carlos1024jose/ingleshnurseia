import {
  classifyGeminiError,
  generateNursingReply,
  getGeminiClient,
  readJsonBody,
  sendJson,
  type AssistantHistoryItem,
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
      error: "API_NOT_CONFIGURED",
      message:
        "No existe una API de inteligencia artificial conectada en este entorno. Verifica GEMINI_API_KEY en Vercel.",
    });
    return;
  }

  const body = await readJsonBody<{ question?: unknown; history?: unknown }>(req);
  const question =
    typeof body?.question === "string" ? body.question.trim() : "";

  if (!question) {
    sendJson(res, 400, {
      error: "EMPTY_QUESTION",
      message: "Por favor escribe una consulta antes de enviar.",
    });
    return;
  }

  if (question.length > 1500) {
    sendJson(res, 400, {
      error: "QUESTION_TOO_LONG",
      message:
        "La consulta excede el límite permitido de 1,500 caracteres. Intenta resumir tu pregunta.",
    });
    return;
  }

  const rawHistory = Array.isArray(body?.history) ? body.history : [];
  const safeHistory: AssistantHistoryItem[] = rawHistory
    .filter(
      (item: unknown): item is AssistantHistoryItem =>
        Boolean(item) &&
        typeof item === "object" &&
        ((item as AssistantHistoryItem).role === "user" ||
          (item as AssistantHistoryItem).role === "assistant") &&
        typeof (item as AssistantHistoryItem).content === "string" &&
        (item as AssistantHistoryItem).content.trim().length > 0
    )
    .slice(-6);

  try {
    const { reply, modelUsed } = await generateNursingReply(
      ai,
      question,
      safeHistory
    );
    if (!reply) {
      sendJson(res, 502, {
        error: "EMPTY_MODEL_RESPONSE",
        message:
          "El servicio de inteligencia artificial no devolvió contenido. Intenta reformular tu consulta.",
      });
      return;
    }

    sendJson(res, 200, { reply, model: modelUsed });
  } catch (error) {
    console.error("Vercel Gemini API error:", error);
    const info = classifyGeminiError(error);

    if (info.isAuthError) {
      sendJson(res, 401, {
        error: "AI_AUTH_ERROR",
        message:
          "La clave GEMINI_API_KEY configurada no es válida o no tiene permisos de acceso.",
      });
      return;
    }

    if (info.isRateLimit) {
      sendJson(res, 429, {
        error: "AI_RATE_LIMIT",
        message:
          "Se alcanzó temporalmente el límite de consultas del servicio de IA. Espera unos segundos e inténtalo de nuevo.",
      });
      return;
    }

    if (info.isModelNotFound) {
      sendJson(res, 502, {
        error: "AI_MODEL_NOT_FOUND",
        message:
          "Ningún modelo de Gemini compatible estuvo disponible para tu clave API (error 404). Revisa la configuración de GEMINI_MODEL o los permisos del proyecto en Google AI Studio.",
      });
      return;
    }

    sendJson(res, 502, {
      error: "AI_UPSTREAM_ERROR",
      message:
        "Ocurrió un error al consultar el servicio de inteligencia artificial. Por favor intenta nuevamente.",
    });
  }
}
