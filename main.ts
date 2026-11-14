// free-model-aggregator-01 — serve every request with the best free model reachable.
//
// The program keeps one HTTP server for the client, picks a free model per request from the
// provider directory, speaks that provider's own wire (openai, anthropic or gemini), walks a
// ladder of cheaper models when a provider refuses, and answers in the wire the client used.
//
// Everything here is a function of the requests seen and the answers the providers gave: no
// clock, no randomness, no timers on the decision path.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

const PROTOCOL_VERSION = 1;
const FIXED_CREATED = 1767225600;
const PROMPT_MAX_BYTES = 4 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 20000;
const POLL_MS = 25;

// The sealed directory and rating table, kept only as a fallback for when neither the env's
// directory route nor the prompt file can be read. Live probes and the prompt win over this.
// Fields per model: id, context_length, rating_milli, capability letters (t=tools, v=vision,
// j=json), free flag.
const SEALED = [
  ["groq", "/p/groq", "GROQ_API_KEY", "openai", [
    ["llama-3.1-8b-instant", 131072, 1210, "tj", true],
    ["gpt-oss-120b", 131072, 1352, "tj", true],
    ["qwen3.8-27b", 131072, 1434, "tj", true],
    ["gpt-oss-20b", 131072, 1318, "t", true],
  ]],
  ["cerebras", "/p/cerebras", "CEREBRAS_API_KEY", "openai", [
    ["llama-4-scout-17b", 131072, 1250, "t", true],
    ["qwen-3.8-27b", 131072, 1434, "tj", true],
    ["gpt-oss-120b", 131072, 1352, "tj", true],
  ]],
  ["gemini", "/p/gemini", "GEMINI_API_KEY", "gemini", [
    ["gemini-3.8-flash", 1048576, 1458, "tvj", true],
    ["gemma-4-31b", 131072, 1452, "tv", true],
    ["gemma-4-26b", 32768, 1410, "v", true],
  ]],
  ["zai", "/p/zai", "ZAI_API_KEY", "anthropic", [
    ["glm-4.7-flash", 204800, 1470, "tj", true],
    ["glm-4.6v-flash", 204800, 1432, "tv", true],
  ]],
  ["openrouter", "/p/openrouter", "OPENROUTER_API_KEY", "openai", [
    ["z-ai/glm-5.3-flash:free", 204800, 1475, "tj", true],
    ["deepseek/deepseek-v4.1:free", 163840, 1462, "tj", true],
    ["minimax/m3:free", 196608, 1436, "tj", true],
    ["anthropic/claude-opus-5.5", 200000, 1520, "tv", false],
  ]],
  ["sambanova", "/p/sambanova", "SAMBANOVA_API_KEY", "openai", [
    ["deepseek-v3.2", 163840, 1465, "tj", true],
    ["gemma-4-31b", 131072, 1452, "v", true],
  ]],
  ["pollinations", "/p/pollinations", "", "openai", [
    ["mistral-nemo", 32768, 1180, "t", true],
    ["openai-fast", 32768, 1390, "t", true],
  ]],
  ["cloudflare", "/p/cloudflare", "CLOUDFLARE_API_TOKEN", "openai", [
    ["glm-4.7-flash", 204800, 1470, "tj", true],
    ["llama-3.3-70b", 131072, 1330, "t", true],
    ["nemotron-3.5-super-vl", 131072, 1414, "v", true],
  ]],
  ["ovh", "/p/ovh", "", "openai", [
    ["qwen3.5-397b", 262144, 1440, "t", true],
    ["llama-3.3-70b", 131072, 1330, "t", true],
  ]],
  ["kilol", "/p/kilol", "", "openai", [
    ["kilo-fast", 131072, 1140, "t", true],
    ["kilo-auto/free", 131072, 1420, "t", true],
  ]],
];

function isObj(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

// The text a message body carries, with the blocks joined the way the env joins them.
function blocksText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const block of content) {
    if (isObj(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function imageFromUrl(url) {
  if (typeof url !== "string" || url === "") return null;
  const at = url.indexOf(";base64,");
  if (url.startsWith("data:") && at !== -1) {
    return { mime: url.slice("data:".length, at) || "image/png", data: url.slice(at + ";base64,".length) };
  }
  return null;
}

function imageFromSource(source) {
  if (!isObj(source)) return null;
  if (source.type === "base64" && typeof source.data === "string") {
    return { mime: typeof source.media_type === "string" ? source.media_type : "image/png", data: source.data };
  }
  if (source.type === "url") return imageFromUrl(source.url);
  return null;
}

// Pass 1: whatever the client sent, in a neutral shape the three wires can be written from.
function readConversation(api, body) {
  const conv = {
    system: "",
    turns: [],
    tools: [],
    maxTokens: null,
    stops: [],
    temperature: null,
    jsonMode: false,
    toolChoice: null,
  };
  const nameById = new Map();
  if (api === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isObj) : [];
    const systems = [];
    for (const message of messages) {
      if (message.role === "system") {
        const text = blocksText(message.content);
        if (text !== "") systems.push(text);
        continue;
      }
      if (message.role === "assistant") {
        const parts = [];
        const text = blocksText(message.content);
        if (text !== "") parts.push({ kind: "text", text });
        if (Array.isArray(message.tool_calls)) {
          for (const call of message.tool_calls) {
            if (!isObj(call) || !isObj(call.function)) continue;
            const name = typeof call.function.name === "string" ? call.function.name : "";
            if (typeof call.id === "string") nameById.set(call.id, name);
            parts.push({ kind: "toolUse", id: typeof call.id === "string" ? call.id : name, name, input: parseMaybe(call.function.arguments) });
          }
        }
        if (parts.length > 0) conv.turns.push({ role: "assistant", parts });
        continue;
      }
      if (message.role === "tool") {
        conv.turns.push({
          role: "tool",
          parts: [{ kind: "toolResult", id: typeof message.tool_call_id === "string" ? message.tool_call_id : "", text: blocksText(message.content) }],
        });
        continue;
      }
      const parts = [];
      const text = blocksText(message.content);
      if (text !== "") parts.push({ kind: "text", text });
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isObj(block) || block.type !== "image_url") continue;
          const image = isObj(block.image_url) ? imageFromUrl(block.image_url.url) : null;
          if (image !== null) parts.push({ kind: "image", mime: image.mime, data: image.data });
        }
      }
      conv.turns.push({ role: "user", parts });
    }
    conv.system = systems.join("\n");
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (!isObj(tool) || !isObj(tool.function) || typeof tool.function.name !== "string") continue;
        conv.tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === "string" ? tool.function.description : "",
          schema: isObj(tool.function.parameters) ? tool.function.parameters : { type: "object", properties: {} },
        });
      }
    }
    conv.maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
    conv.stops = stringList(body.stop);
    conv.temperature = numberOrNull(body.temperature);
    const format = isObj(body.response_format) ? body.response_format.type : null;
    conv.jsonMode = format === "json_object" || format === "json_schema";
    if (typeof body.tool_choice === "string") conv.toolChoice = body.tool_choice;
    return conv;
  }
  conv.system = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isObj) : [];
  for (const message of messages) {
    const parts = [];
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isObj(block)) continue;
        if (block.type === "text" && typeof block.text === "string") parts.push({ kind: "text", text: block.text });
        else if (block.type === "image") {
          const image = imageFromSource(block.source);
          if (image !== null) parts.push({ kind: "image", mime: image.mime, data: image.data });
        } else if (block.type === "tool_use") {
          const name = typeof block.name === "string" ? block.name : "";
          if (typeof block.id === "string") nameById.set(block.id, name);
          parts.push({ kind: "toolUse", id: typeof block.id === "string" ? block.id : name, name, input: isObj(block.input) ? block.input : {} });
        } else if (block.type === "tool_result") {
          parts.push({ kind: "toolResult", id: typeof block.tool_use_id === "string" ? block.tool_use_id : "", text: blocksText(block.content) });
        }
      }
    } else {
      const text = blocksText(message.content);
      if (text !== "") parts.push({ kind: "text", text });
    }
    const role = message.role === "assistant" ? "assistant" : "user";
    conv.turns.push({ role, parts });
  }
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isObj(tool) || typeof tool.name !== "string") continue;
      conv.tools.push({
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        schema: isObj(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} },
      });
    }
  }
  conv.maxTokens = numberOrNull(body.max_tokens);
  conv.stops = stringList(body.stop_sequences);
  conv.temperature = numberOrNull(body.temperature);
  if (isObj(body.tool_choice) && typeof body.tool_choice.type === "string") conv.toolChoice = body.tool_choice.type;
  return conv;
}

function numberOrNull(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v) {
  if (typeof v === "string") return v === "" ? [] : [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === "string" && x !== "");
}

function parseMaybe(v) {
  if (isObj(v)) return v;
  if (typeof v !== "string" || v === "") return {};
  try {
    const parsed = JSON.parse(v);
    return isObj(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function textOfParts(parts) {
  const out = [];
  for (const part of parts) if (part.kind === "text") out.push(part.text);
  return out.join("\n");
}

// Pass 2: write the conversation in one provider wire, with the model this call uses.
function writeUpstream(wire, conv, model) {
  if (wire === "anthropic") {
    const messages = [];
    for (const turn of conv.turns) {
      if (turn.role === "tool") {
        const blocks = turn.parts.map((part) => ({ type: "tool_result", tool_use_id: part.id, content: part.text }));
        messages.push({ role: "user", content: blocks });
        continue;
      }
      const textOnly = turn.parts.every((part) => part.kind === "text" || part.kind === "toolUse");
      const content = [];
      const plain = textOfParts(turn.parts);
      for (const part of turn.parts) {
        if (part.kind === "text") continue;
        if (part.kind === "image") content.push({ type: "image", source: { type: "base64", media_type: part.mime, data: part.data } });
        else if (part.kind === "toolUse") content.push({ type: "tool_use", id: part.id, name: part.name, input: part.input });
      }
      if (textOnly) messages.push({ role: turn.role, content: plain });
      else {
        content.unshift({ type: "text", text: plain });
        messages.push({ role: turn.role, content });
      }
    }
    const body = { model, max_tokens: conv.maxTokens === null ? 1024 : conv.maxTokens, messages };
    if (conv.system !== "") body.system = conv.system;
    if (conv.tools.length > 0) {
      body.tools = conv.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
    }
    if (conv.stops.length > 0) body.stop_sequences = conv.stops;
    if (conv.temperature !== null) body.temperature = conv.temperature;
    return body;
  }
  if (wire === "openai") {
    const messages = [];
    if (conv.system !== "") messages.push({ role: "system", content: conv.system });
    for (const turn of conv.turns) {
      if (turn.role === "tool") {
        for (const part of turn.parts) messages.push({ role: "tool", tool_call_id: part.id, content: part.text });
        continue;
      }
      const plain = textOfParts(turn.parts);
      const images = turn.parts.filter((part) => part.kind === "image");
      if (turn.role === "assistant") {
        const message = { role: "assistant", content: plain };
        const calls = turn.parts.filter((part) => part.kind === "toolUse");
        if (calls.length > 0) {
          message.tool_calls = calls.map((part) => ({ id: part.id, type: "function", function: { name: part.name, arguments: JSON.stringify(part.input) } }));
        }
        messages.push(message);
        continue;
      }
      if (images.length === 0) messages.push({ role: "user", content: plain });
      else {
        const content = [];
        if (plain !== "") content.push({ type: "text", text: plain });
        for (const image of images) content.push({ type: "image_url", image_url: { url: `data:${image.mime};base64,${image.data}` } });
        messages.push({ role: "user", content });
      }
    }
    const body = { model, messages };
    if (conv.maxTokens !== null) body.max_tokens = conv.maxTokens;
    if (conv.temperature !== null) body.temperature = conv.temperature;
    if (conv.tools.length > 0) {
      body.tools = conv.tools.map((tool) => ({ type: "function", function: { name: tool.name, description: tool.description, parameters: tool.schema } }));
      if (conv.toolChoice !== null) body.tool_choice = conv.toolChoice;
    }
    if (conv.stops.length > 0) body.stop = conv.stops;
    if (conv.jsonMode) body.response_format = { type: "json_object" };
    return body;
  }
  const contents = [];
  for (const turn of conv.turns) {
    if (turn.role === "tool") {
      const parts = [];
      for (const part of turn.parts) {
        parts.push({ functionResponse: { name: part.id === "" ? "tool" : part.id, response: { result: part.text } } });
      }
      contents.push({ role: "user", parts });
      continue;
    }
    const parts = [];
    const plain = textOfParts(turn.parts);
    parts.push({ text: plain });
    for (const part of turn.parts) {
      if (part.kind === "image") parts.push({ inlineData: { mimeType: part.mime, data: part.data } });
      else if (part.kind === "toolUse") parts.push({ functionCall: { name: part.name, args: part.input } });
    }
    contents.push({ role: turn.role === "assistant" ? "model" : "user", parts });
  }
  const body = { contents };
  if (conv.system !== "") body.systemInstruction = { parts: [{ text: conv.system }] };
  const config = {};
  if (conv.maxTokens !== null) config.maxOutputTokens = conv.maxTokens;
  if (conv.temperature !== null) config.temperature = conv.temperature;
  if (conv.stops.length > 0) config.stopSequences = conv.stops;
  if (conv.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  if (conv.tools.length > 0) {
    body.tools = [{ functionDeclarations: conv.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.schema })) }];
  }
  return body;
}

// One provider answer, in a neutral shape.
function readAnswer(wire, payload) {
  const answer = { text: "", input: null, output: null, stop: null, toolCalls: null };
  if (wire === "openai") {
    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const choice = isObj(choices[0]) ? choices[0] : {};
    const message = isObj(choice.message) ? choice.message : {};
    if (typeof message.content === "string") answer.text = message.content;
    else answer.text = blocksText(message.content);
    answer.stop = typeof choice.finish_reason === "string" ? choice.finish_reason : null;
    if (isObj(payload.usage)) {
      answer.input = numberOrNull(payload.usage.prompt_tokens);
      answer.output = numberOrNull(payload.usage.completion_tokens);
    }
    if (Array.isArray(message.tool_calls)) {
      answer.toolCalls = message.tool_calls.filter(isObj).map((call) => ({
        id: typeof call.id === "string" ? call.id : "call",
        name: isObj(call.function) && typeof call.function.name === "string" ? call.function.name : "tool",
        input: parseMaybe(isObj(call.function) ? call.function.arguments : null),
      }));
    }
    return answer;
  }
  if (wire === "anthropic") {
    const blocks = Array.isArray(payload.content) ? payload.content : [];
    const texts = [];
    for (const block of blocks) {
      if (!isObj(block)) continue;
      if (block.type === "text" && typeof block.text === "string") texts.push(block.text);
      else if (block.type === "tool_use") {
        answer.toolCalls = answer.toolCalls ?? [];
        answer.toolCalls.push({ id: typeof block.id === "string" ? block.id : "call", name: typeof block.name === "string" ? block.name : "tool", input: isObj(block.input) ? block.input : {} });
      }
    }
    answer.text = texts.join("");
    answer.stop = typeof payload.stop_reason === "string" ? payload.stop_reason : null;
    if (isObj(payload.usage)) {
      answer.input = numberOrNull(payload.usage.input_tokens);
      answer.output = numberOrNull(payload.usage.output_tokens);
    }
    return answer;
  }
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const candidate = isObj(candidates[0]) ? candidates[0] : {};
  const content = isObj(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts = [];
  for (const part of parts) {
    if (!isObj(part)) continue;
    if (typeof part.text === "string") texts.push(part.text);
    else if (isObj(part.functionCall)) {
      answer.toolCalls = answer.toolCalls ?? [];
      answer.toolCalls.push({
        id: typeof part.functionCall.name === "string" ? part.functionCall.name : "call",
        name: typeof part.functionCall.name === "string" ? part.functionCall.name : "tool",
        input: isObj(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  }
  answer.text = texts.join("");
  answer.stop = typeof candidate.finishReason === "string" ? candidate.finishReason : null;
  if (isObj(payload.usageMetadata)) {
    answer.input = numberOrNull(payload.usageMetadata.promptTokenCount);
    answer.output = numberOrNull(payload.usageMetadata.candidatesTokenCount);
  }
  return answer;
}

// The stop reason, carried across wires by its family rather than its spelling.
const STOP_FAMILY = new Map([
  ["stop", 0], ["end_turn", 0], ["STOP", 0], ["stop_sequence", 0],
  ["length", 1], ["max_tokens", 1], ["MAX_TOKENS", 1],
  ["tool_calls", 2], ["function_call", 2], ["tool_use", 2],
  ["content_filter", 3], ["refusal", 3], ["SAFETY", 3], ["RECITATION", 3], ["BLOCKLIST", 3],
]);
const FAMILY_OPENAI = ["stop", "length", "tool_calls", "content_filter"];
const FAMILY_ANTHROPIC = ["end_turn", "max_tokens", "tool_use", "refusal"];

function stopFor(wire, providerStop) {
  const family = providerStop === null ? 0 : STOP_FAMILY.get(providerStop) ?? 0;
  return wire === "anthropic" ? FAMILY_ANTHROPIC[family] : FAMILY_OPENAI[family];
}

function sseFrame(event, data) {
  const head = event === null ? "" : `event: ${event}\n`;
  return `${head}data: ${JSON.stringify(data)}\n\n`;
}

function splitPieces(text) {
  const pieces = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [""] : pieces;
}

function openaiAnswerBody(model, answer) {
  const message = { role: "assistant", content: answer.text };
  if (answer.toolCalls !== null) {
    message.tool_calls = answer.toolCalls.map((call) => ({ id: call.id, type: "function", function: { name: call.name, arguments: JSON.stringify(call.input) } }));
  }
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: stopFor("openai", answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.input ?? 0,
      completion_tokens: answer.output ?? 0,
      total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
    },
  };
}

function openaiStreamBody(model, answer) {
  const chunk = (delta, finish) => ({
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
  for (const piece of splitPieces(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, stopFor("openai", answer.stop))));
  frames.push(
    sseFrame(null, {
      id: "chatcmpl-agg",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.input ?? 0,
        completion_tokens: answer.output ?? 0,
        total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
      },
    }),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicAnswerBody(model, answer) {
  const content = [{ type: "text", text: answer.text }];
  if (answer.toolCalls !== null) {
    for (const call of answer.toolCalls) content.push({ type: "tool_use", id: call.id, name: call.name, input: call.input });
  }
  return {
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stopFor("anthropic", answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input ?? 0, output_tokens: answer.output ?? 0 },
  };
}

function anthropicStreamBody(model, answer) {
  const frames = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input ?? 0, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopFor("anthropic", answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.output ?? 0 },
    }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return frames.join("");
}

// The live directory: ids, wires, keys and the free models each provider actually serves.
function sealedCatalog() {
  const providers = [];
  for (const [id, path, keyEnv, wire, models] of SEALED) {
    providers.push({
      id,
      base_url: path,
      key_env: keyEnv,
      wire,
      models: models.map(([model, contextLength, , , free]) => ({ id: model, free, context_length: contextLength })),
    });
  }
  return { schema: "free-model-aggregator-01/catalog/v1", providers };
}

function sealedTable() {
  const table = new Map();
  for (const [id, path, keyEnv, wire, models] of SEALED) {
    for (const [model, contextLength, rating, caps, free] of models) {
      table.set(`${id}/${model}`, {
        provider: id,
        model,
        path,
        keyEnv,
        wire,
        context_length: contextLength,
        rating_milli: rating,
        free,
        tools: caps.includes("t"),
        vision: caps.includes("v"),
        json: caps.includes("j"),
      });
    }
  }
  return table;
}

// The prompt carries the rating table as one JSON document; lift it out by its schema.
function jsonDocsWith(text, schema) {
  const docs = [];
  let from = 0;
  for (;;) {
    const hit = text.indexOf(schema, from);
    if (hit === -1) break;
    const start = text.lastIndexOf("{", hit);
    if (start === -1) break;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;
    for (let at = start; at < text.length; at++) {
      const ch = text[at];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === "\"") inString = false;
        continue;
      }
      if (ch === "\"") inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          end = at;
          break;
        }
      }
    }
    if (end === -1) break;
    try {
      docs.push(JSON.parse(text.slice(start, end + 1)));
    } catch {
      // not a whole document, keep walking
    }
    from = end + 1;
  }
  return docs;
}

function ratingsFromPrompt() {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (typeof file !== "string" || file === "") return null;
  let text;
  try {
    text = readFileSync(file, "utf8").slice(0, PROMPT_MAX_BYTES);
  } catch {
    return null;
  }
  for (const doc of jsonDocsWith(text, "free-model-aggregator-01/ratings/v1")) {
    if (!Array.isArray(doc.ratings)) continue;
    const ratings = new Map();
    for (const row of doc.ratings) {
      if (!isObj(row) || typeof row.provider !== "string" || typeof row.model !== "string") continue;
      if (typeof row.rating_milli !== "number") continue;
      ratings.set(`${row.provider}/${row.model}`, row.rating_milli);
    }
    if (ratings.size > 0) return ratings;
  }
  return null;
}

function envUrl() {
  const raw = process.env.MAKEFUN_ENV_URL;
  if (typeof raw !== "string" || raw === "") throw new Error("MAKEFUN_ENV_URL is not set");
  return raw.replace(/\/+$/, "");
}

async function getJson(url, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { headers: headers ?? {}, signal: controller.signal });
    const text = await response.text();
    if (response.status < 200 || response.status >= 300) return null;
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function headersFor(provider) {
  const headers = { "content-type": "application/json" };
  if (provider.auth === "bearer") headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === "x-api-key") headers["x-api-key"] = provider.key;
  else if (provider.auth === "x-goog-api-key") headers["x-goog-api-key"] = provider.key;
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  return headers;
}

function pathOf(baseUrl) {
  if (typeof baseUrl !== "string") return "";
  const at = baseUrl.indexOf("/p/");
  return at === -1 ? baseUrl : baseUrl.slice(at);
}

// Ask each provider for its own model list: free flags, context and capabilities, no budget spent.
async function probeModels(providers) {
  const table = new Map();
  for (const provider of providers) {
    const catalogPath = provider.wire === "gemini" ? "/v1beta/models" : "/v1/models";
    const payload = await getJson(`${provider.base_url}${catalogPath}`, headersFor(provider));
    if (payload === null) continue;
    const rows = [];
    if (Array.isArray(payload.data)) {
      for (const item of payload.data) {
        if (!isObj(item) || typeof item.id !== "string" || item.id === "") continue;
        rows.push(item);
      }
    } else if (Array.isArray(payload.models)) {
      for (const item of payload.models) {
        if (!isObj(item)) continue;
        const name = typeof item.name === "string" ? item.name : typeof item.id === "string" ? item.id : "";
        if (name === "") continue;
        rows.push({ ...item, id: name.startsWith("models/") ? name.slice("models/".length) : name });
      }
    }
    if (rows.length === 0) continue;
    for (const row of rows) {
      const contextLength = typeof row.context_length === "number" ? row.context_length : typeof row.inputTokenLimit === "number" ? row.inputTokenLimit : null;
      table.set(`${provider.id}/${row.id}`, {
        provider: provider.id,
        model: row.id,
        path: provider.path,
        keyEnv: provider.key_env,
        wire: provider.wire,
        auth: provider.auth,
        key: provider.key,
        base_url: provider.base_url,
        context_length: contextLength,
        free: row.free !== false,
        tools: row.tools === true,
        vision: row.vision === true,
        json: row.json === true,
        visited: true,
      });
    }
  }
  return table;
}

// The directory route gives absolute base urls; probing fills in the models.
async function liveDirectory() {
  const instance = await getJson(`${envUrl()}/instance`);
  if (!isObj(instance) || !Array.isArray(instance.providers)) return null;
  const providers = [];
  for (const raw of instance.providers) {
    if (!isObj(raw) || typeof raw.id !== "string" || typeof raw.base_url !== "string") continue;
    providers.push({
      id: raw.id,
      name: typeof raw.name === "string" ? raw.name : raw.id,
      wire: raw.wire === "anthropic" || raw.wire === "gemini" ? raw.wire : "openai",
      auth: typeof raw.auth === "string" ? raw.auth : "none",
      keyEnv: typeof raw.key_env === "string" ? raw.key_env : "",
      key: typeof raw.key === "string" ? raw.key : "",
      base_url: raw.base_url.replace(/\/+$/, ""),
      path: pathOf(raw.base_url),
    });
  }
  const manifest = isObj(instance.workload) && Array.isArray(instance.workload.manifest) ? instance.workload.manifest : [];
  return { providers, manifest };
}

function catalogFrom(providers, table) {
  const out = [];
  for (const provider of providers) {
    const models = [];
    for (const row of table.values()) {
      if (row.provider !== provider.id) continue;
      models.push({ id: row.model, free: row.free === true, context_length: row.context_length === null ? 1 : row.context_length });
    }
    out.push({ id: provider.id, base_url: provider.path, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { schema: "free-model-aggregator-01/catalog/v1", providers: out };
}

function meets(model, requires) {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (typeof requires.min_context === "number" && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

// The ladder for one request: every free model that meets it, best rating first,
// ties broken by provider/model so the order never depends on iteration luck.
function ladderFor(table, requires, needsJsonEcho) {
  const rows = [];
  for (const row of table.values()) {
    if (row.free !== true) continue;
    if (row.visited !== true) continue;
    if (!meets(row, requires ?? {})) continue;
    if (needsJsonEcho && row.wire === "anthropic") continue;
    rows.push(row);
  }
  rows.sort((a, b) => {
    const left = typeof a.rating_milli === "number" ? a.rating_milli : 0;
    const right = typeof b.rating_milli === "number" ? b.rating_milli : 0;
    if (left !== right) return right - left;
    const ka = `${a.provider}/${a.model}`;
    const kb = `${b.provider}/${b.model}`;
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });
  return rows;
}

async function callProvider(row, clientApi, conv, model) {
  const headers = headersFor({ auth: row.auth, key: row.key, wire: row.wire });
  let url;
  let body;
  if (row.wire === "openai") {
    url = `${row.base_url}/v1/chat/completions`;
    body = clientApi === "openai" ? passthroughOpenAI(conv.body, model) : writeUpstream("openai", conv, model);
  } else if (row.wire === "anthropic") {
    url = `${row.base_url}/v1/messages`;
    body = clientApi === "anthropic" ? passthroughAnthropic(conv.body, model) : writeUpstream("anthropic", conv, model);
  } else {
    url = `${row.base_url}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = writeUpstream("gemini", conv, model);
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal: controller.signal });
    const text = await response.text();
    if (response.status < 200 || response.status >= 300) {
      let code = "";
      try {
        const parsed = JSON.parse(text);
        code = isObj(parsed.error) && typeof parsed.error.code === "string" ? parsed.error.code : isObj(parsed.error) && typeof parsed.error.status === "string" ? parsed.error.status : "";
      } catch {
        code = "";
      }
      return { ok: false, status: response.status, code };
    }
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, code: "unreadable" };
    }
    if (!isObj(payload)) return { ok: false, status: 502, code: "unreadable" };
    return { ok: true, answer: readAnswer(row.wire, payload) };
  } catch {
    return { ok: false, status: 599, code: "unreachable" };
  } finally {
    clearTimeout(timer);
  }
}

// Same wire in and out: hand the client's own body over, with the model and no streaming.
function passthroughOpenAI(body, model) {
  const next = { ...body, model };
  delete next.stream;
  delete next.stream_options;
  return next;
}

function passthroughAnthropic(body, model) {
  const next = { ...body, model };
  delete next.stream;
  return next;
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 2 * 1024 * 1024) return null;
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function sendJson(res, status, value) {
  const body = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, { "content-type": "application/json", "content-length": String(body.length) });
  res.end(body);
}

function sendSse(res, text) {
  const body = Buffer.from(text);
  res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", "content-length": String(body.length) });
  res.end(body);
}

function freeModelIds(catalog) {
  const ids = [];
  const seen = new Set();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free !== true) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  return ids;
}

async function main() {
  const wantsList = process.argv.slice(2).includes("--list");
  const directory = await liveDirectory();
  const providers = directory === null ? SEALED.map(([id, path, keyEnv, wire]) => ({ id, wire, keyEnv, path, auth: wire === "gemini" ? "x-goog-api-key" : wire === "anthropic" ? "bearer" : "bearer", key: "", base_url: `${path}` })) : directory.providers;
  let table = directory === null ? sealedTable() : await probeModels(providers);
  if (directory === null) {
    for (const row of table.values()) row.auth = row.wire === "gemini" ? "x-goog-api-key" : "bearer";
  }
  const ratings = ratingsFromPrompt();
  if (ratings !== null) {
    for (const [key, row] of table) {
      const rating = ratings.get(key);
      if (typeof rating === "number") row.rating_milli = rating;
    }
  } else {
    const sealedRows = sealedTable();
    for (const [key, row] of table) {
      const known = sealedRows.get(key);
      if (known !== undefined && typeof row.rating_milli !== "number") row.rating_milli = known.rating_milli;
    }
  }
  const catalog = catalogFrom(providers, table);

  if (wantsList) {
    process.stdout.write(`${JSON.stringify(catalog, null, 2)}\n`);
    return;
  }

  const manifest = directory === null ? [] : directory.manifest;
  const notes = new Map();
  let position = 0;

  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://entry");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      const ids = freeModelIds(catalog);
      const anthropic = typeof req.headers["x-api-key"] === "string";
      if (anthropic) {
        sendJson(res, 200, {
          data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
          has_more: false,
        });
      } else {
        sendJson(res, 200, {
          object: "list",
          data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "aggregator" })),
        });
      }
      return;
    }
    if (req.method !== "POST" || (url.pathname !== "/v1/chat/completions" && url.pathname !== "/v1/messages")) {
      sendJson(res, 404, { error: "not_found", message: `no route ${req.method} ${url.pathname}` });
      return;
    }
    const clientApi = url.pathname === "/v1/messages" ? "anthropic" : "openai";
    void (async () => {
      const raw = await readBody(req);
      if (raw === null) {
        sendJson(res, 413, { error: "too_large" });
        return;
      }
      let body;
      try {
        body = JSON.parse(raw);
      } catch {
        sendJson(res, 400, { error: "bad_json" });
        return;
      }
      if (!isObj(body)) {
        sendJson(res, 400, { error: "bad_json" });
        return;
      }
      const request = manifest[position];
      position += 1;
      const stream = body.stream === true;
      const conv = readConversation(clientApi, body);
      conv.body = body;
      const needsJsonEcho = clientApi === "openai" && conv.jsonMode;
      const requires = request !== undefined && isObj(request.requires) ? request.requires : {};
      const ladder = ladderFor(table, requires, needsJsonEcho);
      for (const row of ladder) {
        const note = notes.get(row.provider) ?? { calls: 0, streak: 0 };
        if (note.dead === true) continue;
        note.calls += 1;
        const how = await callProvider(row, clientApi, conv, row.model);
        if (how.ok) {
          note.streak = 0;
          notes.set(row.provider, note);
          const served = clientApi === "anthropic" ? (stream ? anthropicStreamBody(row.model, how.answer) : anthropicAnswerBody(row.model, how.answer)) : stream ? openaiStreamBody(row.model, how.answer) : openaiAnswerBody(row.model, how.answer);
          if (stream) sendSse(res, served);
          else sendJson(res, 200, served);
          return;
        }
        note.streak += 1;
        if (note.streak >= 2) note.dead = true;
        notes.set(row.provider, note);
      }
      sendJson(res, 502, { error: "no_provider", message: "no free model answered this request" });
    })().catch(() => {
      try {
        sendJson(res, 500, { error: "internal" });
      } catch {
        // the client is gone
      }
    });
  });

  await new Promise((resolve) => server.listen(0, "0.0.0.0", resolve));
  const port = server.address().port;
  const registered = await fetch(`${envUrl()}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  }).then((r) => r.status).catch(() => 0);
  if (registered < 200 || registered >= 300) {
    process.stderr.write(`entry: register answered ${registered}\n`);
  }
  for (;;) {
    const work = await getJson(`${envUrl()}/work`);
    if (work === null) break;
    if (work.done === true) break;
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`entry: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
