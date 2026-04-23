#!/usr/bin/env node

import { createHash } from "crypto";
import { execFile } from "child_process";
import { existsSync } from "fs";
import { appendFile, mkdir, readFile, writeFile } from "fs/promises";
import { basename, dirname, join, resolve } from "path";
import { promisify } from "util";

const execFileAsync = promisify(execFile);
const DEFAULT_STATE_FILE = "feishu-human-intervention-state.json";
const DEFAULT_LOG_FILE = "feishu-human-intervention.jsonl";
const FINGERPRINT_TTL_MS = 12 * 60 * 60 * 1000;
const MAX_RECENT_FINGERPRINTS = 200;
const DELEGATE_TIMEOUT_MS = 20_000;
const FEISHU_TIMEOUT_MS = 10_000;

function safeString(value) {
  return typeof value === "string" ? value : "";
}

function normalizeWhitespace(value) {
  return safeString(value).replace(/\s+/g, " ").trim();
}

function truncate(value, maxLength = 280) {
  const text = normalizeWhitespace(value);
  if (!text) {
    return "";
  }
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function firstNonEmpty(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function readPayloadArg(argv) {
  if (argv.length === 0) {
    return "";
  }
  const candidate = argv[argv.length - 1];
  if (!candidate || !candidate.trim().startsWith("{")) {
    return "";
  }
  return candidate;
}

function parseArgs(argv) {
  const rawPayload = readPayloadArg(argv);
  const optionArgs = rawPayload ? argv.slice(0, -1) : argv.slice();
  const options = {
    delegate: "",
    feishuWebhook: process.env.FEISHU_BOT_WEBHOOK || "",
  };

  for (let index = 0; index < optionArgs.length; index += 1) {
    const arg = optionArgs[index];
    const next = optionArgs[index + 1];
    if (arg === "--delegate" && next) {
      options.delegate = next;
      index += 1;
      continue;
    }
    if (arg === "--feishu-webhook" && next) {
      options.feishuWebhook = next;
      index += 1;
      continue;
    }
  }

  return { options, rawPayload };
}

function parseJson(raw) {
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function resolveCwd(payload) {
  return resolve(
    firstNonEmpty(
      safeString(payload?.cwd),
      safeString(payload?.projectPath),
      process.cwd(),
    ),
  );
}

function resolveSessionId(payload) {
  return firstNonEmpty(
    safeString(payload?.sessionId),
    safeString(payload?.session_id),
    safeString(payload?.["session-id"]),
  );
}

function resolveThreadId(payload) {
  return firstNonEmpty(
    safeString(payload?.thread_id),
    safeString(payload?.["thread-id"]),
  );
}

function resolveInputMessages(payload) {
  const candidates = payload?.input_messages ?? payload?.["input-messages"];
  if (!Array.isArray(candidates)) {
    return [];
  }
  return candidates.map((item) => safeString(item)).filter(Boolean);
}

function resolveLastUserInput(payload) {
  const inputMessages = resolveInputMessages(payload);
  return inputMessages.length > 0 ? inputMessages[inputMessages.length - 1] : "";
}

function resolveAssistantMessage(payload) {
  return firstNonEmpty(
    safeString(payload?.last_assistant_message),
    safeString(payload?.["last-assistant-message"]),
    safeString(payload?.message),
  );
}

function resolveQuestion(payload, assistantMessage) {
  return firstNonEmpty(
    safeString(payload?.question),
    safeString(payload?.user_question),
    safeString(payload?.["user-question"]),
    assistantMessage.endsWith("?") ? assistantMessage : "",
  );
}

function looksLikeHumanIntervention(payload) {
  const event = normalizeWhitespace(payload?.event).toLowerCase();
  const reason = normalizeWhitespace(payload?.reason).toLowerCase();
  const assistantMessage = resolveAssistantMessage(payload);
  const normalizedAssistant = normalizeWhitespace(assistantMessage).toLowerCase();
  const question = resolveQuestion(payload, assistantMessage);

  if (event === "ask-user-question") {
    return {
      matched: true,
      reason: "ask-user-question",
      assistantMessage,
      question,
    };
  }

  if (question) {
    return {
      matched: true,
      reason: "question-present",
      assistantMessage,
      question,
    };
  }

  if (event === "session-idle" && /wait|input|response|approval/.test(reason)) {
    return {
      matched: true,
      reason: event,
      assistantMessage,
      question: "",
    };
  }

  const phrases = [
    "would you like",
    "do you want me to",
    "do you want",
    "should i",
    "shall i",
    "awaiting approval",
    "awaiting your approval",
    "approval before applying",
    "waiting for your response",
    "waiting for input",
    "let me know if",
    "let me know whether",
    "i can also",
    "i could also",
    "say go",
    "say yes",
    "type continue",
  ];

  if (phrases.some((phrase) => normalizedAssistant.includes(phrase))) {
    return {
      matched: true,
      reason: "assistant-needs-input",
      assistantMessage,
      question,
    };
  }

  return {
    matched: false,
    reason: "",
    assistantMessage,
    question,
  };
}

function buildFingerprint(payload, signal) {
  const fingerprintSource = JSON.stringify({
    sessionId: resolveSessionId(payload),
    threadId: resolveThreadId(payload),
    reason: signal.reason,
    question: signal.question,
    assistantMessage: signal.assistantMessage,
  });
  return createHash("sha256").update(fingerprintSource).digest("hex");
}

async function ensureDir(path) {
  await mkdir(path, { recursive: true }).catch(() => {});
}

async function appendJsonl(logPath, entry) {
  await ensureDir(dirname(logPath));
  await appendFile(logPath, `${JSON.stringify(entry)}\n`).catch(() => {});
}

async function readJsonIfExists(path, fallback) {
  if (!existsSync(path)) {
    return fallback;
  }
  try {
    return JSON.parse(await readFile(path, "utf-8"));
  } catch {
    return fallback;
  }
}

async function wasRecentlySent(statePath, fingerprint) {
  const now = Date.now();
  const state = await readJsonIfExists(statePath, { recent: {} });
  const recent = state && typeof state === "object" && state.recent && typeof state.recent === "object"
    ? state.recent
    : {};

  const nextRecent = {};
  for (const [key, timestamp] of Object.entries(recent)) {
    const numericTimestamp = typeof timestamp === "number" ? timestamp : 0;
    if (now - numericTimestamp <= FINGERPRINT_TTL_MS) {
      nextRecent[key] = numericTimestamp;
    }
  }

  const alreadySent = typeof nextRecent[fingerprint] === "number";
  nextRecent[fingerprint] = now;

  const trimmedEntries = Object.entries(nextRecent)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, MAX_RECENT_FINGERPRINTS);

  await ensureDir(dirname(statePath));
  await writeFile(
    statePath,
    JSON.stringify({ recent: Object.fromEntries(trimmedEntries) }, null, 2),
  ).catch(() => {});

  return alreadySent;
}

async function runDelegate(delegateScript, rawPayload, cwd, logPath) {
  if (!delegateScript) {
    return;
  }
  try {
    await execFileAsync(
      "node",
      [delegateScript, rawPayload],
      {
        cwd,
        timeout: DELEGATE_TIMEOUT_MS,
      },
    );
  } catch (error) {
    await appendJsonl(logPath, {
      timestamp: new Date().toISOString(),
      type: "delegate_error",
      delegateScript,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

function buildFeishuMessage(payload, signal, cwd) {
  const sessionId = resolveSessionId(payload) || "unknown";
  const threadId = resolveThreadId(payload) || "unknown";
  const projectName = firstNonEmpty(
    safeString(payload?.projectName),
    basename(cwd),
  );
  const lastUserInput = truncate(resolveLastUserInput(payload), 180);
  const question = truncate(signal.question, 220);
  const assistantMessage = truncate(signal.assistantMessage, 220);
  const lines = [
    "Codex 需要人工介入",
    `项目: ${projectName}`,
    `会话: ${sessionId}`,
    `线程: ${threadId}`,
    `原因: ${signal.reason}`,
  ];

  if (question) {
    lines.push(`问题: ${question}`);
  }
  if (assistantMessage && assistantMessage !== question) {
    lines.push(`最近输出: ${assistantMessage}`);
  }
  if (lastUserInput) {
    lines.push(`上一条用户输入: ${lastUserInput}`);
  }

  lines.push(`时间: ${new Date().toLocaleString("zh-CN", { hour12: false })}`);
  lines.push(`目录: ${cwd}`);

  return lines.join("\n");
}

async function postToFeishu(webhookUrl, text) {
  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      msg_type: "text",
      content: {
        text,
      },
    }),
    signal: AbortSignal.timeout(FEISHU_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(`feishu_http_${response.status}`);
  }
}

async function main() {
  const { options, rawPayload } = parseArgs(process.argv.slice(2));
  const payload = parseJson(rawPayload);

  if (!payload) {
    process.exit(0);
  }

  const cwd = resolveCwd(payload);
  const omxDir = join(cwd, ".omx");
  const statePath = join(omxDir, "state", DEFAULT_STATE_FILE);
  const logPath = join(omxDir, "logs", DEFAULT_LOG_FILE);

  await runDelegate(options.delegate, rawPayload, cwd, logPath);

  const signal = looksLikeHumanIntervention(payload);
  if (!signal.matched) {
    process.exit(0);
  }

  if (!options.feishuWebhook) {
    await appendJsonl(logPath, {
      timestamp: new Date().toISOString(),
      type: "feishu_skipped",
      reason: "missing_webhook",
      sessionId: resolveSessionId(payload) || null,
    });
    process.exit(0);
  }

  const fingerprint = buildFingerprint(payload, signal);
  const duplicated = await wasRecentlySent(statePath, fingerprint);
  if (duplicated) {
    await appendJsonl(logPath, {
      timestamp: new Date().toISOString(),
      type: "feishu_skipped",
      reason: "duplicate",
      sessionId: resolveSessionId(payload) || null,
      fingerprint,
    });
    process.exit(0);
  }

  const text = buildFeishuMessage(payload, signal, cwd);

  try {
    await postToFeishu(options.feishuWebhook, text);
    await appendJsonl(logPath, {
      timestamp: new Date().toISOString(),
      type: "feishu_sent",
      sessionId: resolveSessionId(payload) || null,
      threadId: resolveThreadId(payload) || null,
      reason: signal.reason,
      fingerprint,
    });
  } catch (error) {
    await appendJsonl(logPath, {
      timestamp: new Date().toISOString(),
      type: "feishu_error",
      sessionId: resolveSessionId(payload) || null,
      error: error instanceof Error ? error.message : String(error),
      fingerprint,
    });
  }
}

await main();
