import { performance } from "node:perf_hooks";
import {
  classifyGeminiError,
  getGeminiClient,
  readJsonBody,
  roundMs,
  sendJson,
  streamNursingReply,
  type AssistantHistoryItem,
  type VercelReq,
  type VercelRes,
} from "./_gemini.js";

function writeSseEvent(res: VercelRes, payload: unknown): void {
  res.write?.(`data: ${JSON.stringify(payload)}\n\n`);
}

export default async function handler(req: VercelReq, res: VercelRes) {
  const tReqStart = performance.now();

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

  const body = await readJsonBody<{
    question?: unknown;
    history?: unknown;
    stream?: unknown;
  }>(req);

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
    .slice(-4);

  const wantsStream =
    body?.stream === true ||
    String(req.headers?.accept ?? "").includes("text/event-stream");

  const tPrepDone = performance.now();

  const abortController = new AbortController();
  res.on?.("close", () => {
    if (!res.writableEnded) {
      abortController.abort();
    }
  });

  if (wantsStream && typeof res.write === "function") {
    res.statusCode = 200;
    res.setHeader?.("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader?.("Cache-Control", "no-cache, no-transform");
    res.setHeader?.("Connection", "keep-alive");
    res.setHeader?.("X-Accel-Buffering", "no");
    res.flushHeaders?.();

    let sentMeta = false;

    try {
      const { reply, metrics } = await streamNursingReply({
        ai,
        question,
        safeHistory,
        tReqStart,
        tPrepDone,
        signal: abortController.signal,
        onMeta: (meta) => {
          if (!sentMeta && !abortController.signal.aborted) {
            sentMeta = true;
            writeSseEvent(res, { type: "meta", ...meta });
          }
        },
        onChunk: (text) => {
          if (!abortController.signal.aborted) {
            writeSseEvent(res, { type: "delta", text });
          }
        },
      });

      if (!reply && !abortController.signal.aborted) {
        writeSseEvent(res, {
          type: "error",
          code: "EMPTY_MODEL_RESPONSE",
          message:
            "El servicio de inteligencia artificial no devolvió contenido. Intenta reformular tu consulta.",
        });
        res.end?.();
        return;
      }

      const finalTotalMs = roundMs(performance.now() - tReqStart);
      console.info(
        "[assistant:perf]",
        JSON.stringify({
          runtime: "vercel",
          stream: true,
          ...metrics,
          totalMs: finalTotalMs,
        })
      );

      if (!abortController.signal.aborted) {
        writeSseEvent(res, {
          type: "done",
          model: metrics.modelUsed,
          tier: metrics.tier,
          metrics: { ...metrics, totalMs: finalTotalMs },
        });
      }
      res.end?.();
    } catch (error) {
      if (abortController.signal.aborted) {
        res.end?.();
        return;
      }

      const info = classifyGeminiError(error);
      console.error("[assistant:error]", {
        runtime: "vercel",
        status: info.status,
        isAuthError: info.isAuthError,
        isRateLimit: info.isRateLimit,
        isModelNotFound: info.isModelNotFound,
        elapsedMs: roundMs(performance.now() - tReqStart),
      });

      const code = info.isAuthError
        ? "AI_AUTH_ERROR"
        : info.isRateLimit
        ? "AI_RATE_LIMIT"
        : info.isModelNotFound
        ? "AI_MODEL_NOT_FOUND"
        : "AI_UPSTREAM_ERROR";

      const message = info.isAuthError
        ? "La clave GEMINI_API_KEY configurada no es válida o no tiene permisos de acceso."
        : info.isRateLimit
        ? "Se alcanzó temporalmente el límite de consultas del servicio de IA. Espera unos segundos e inténtalo de nuevo."
        : info.isModelNotFound
        ? "Ningún modelo de Gemini compatible estuvo disponible para tu clave API (error 404)."
        : "Ocurrió un error al consultar el servicio de inteligencia artificial. Por favor intenta nuevamente.";

      writeSseEvent(res, { type: "error", code, message });
      res.end?.();
    }
    return;
  }

  try {
    let bufferedReply = "";
    const { reply, metrics } = await streamNursingReply({
      ai,
      question,
      safeHistory,
      tReqStart,
      tPrepDone,
      signal: abortController.signal,
      onChunk: (delta) => {
        bufferedReply += delta;
      },
    });

    const finalReply = reply || bufferedReply.trim();
    if (!finalReply) {
      sendJson(res, 502, {
        error: "EMPTY_MODEL_RESPONSE",
        message:
          "El servicio de inteligencia artificial no devolvió contenido. Intenta reformular tu consulta.",
      });
      return;
    }

    const finalTotalMs = roundMs(performance.now() - tReqStart);
    console.info(
      "[assistant:perf]",
      JSON.stringify({
        runtime: "vercel",
        stream: false,
        ...metrics,
        totalMs: finalTotalMs,
      })
    );

    sendJson(res, 200, {
      reply: finalReply,
      model: metrics.modelUsed,
      tier: metrics.tier,
      metrics: { ...metrics, totalMs: finalTotalMs },
    });
  } catch (error) {
    const info = classifyGeminiError(error);
    console.error("[assistant:error]", {
      runtime: "vercel",
      status: info.status,
      isAuthError: info.isAuthError,
      isRateLimit: info.isRateLimit,
      isModelNotFound: info.isModelNotFound,
      elapsedMs: roundMs(performance.now() - tReqStart),
    });

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
