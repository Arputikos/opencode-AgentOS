import { asSchema, type ModelMessage, type Tool } from "ai"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { Token } from "@/util/token"

/**
 * Patched (agentos): what a request to the model is made of, in tokens.
 *
 * Every enabled MCP tool travels with every request, schema and all, and a
 * single hosted MCP server can outweigh everything else put together (one
 * production server exposed 120 tools worth ~300k tokens). When such a request
 * does not fit, the provider either rejects it or ends the turn with nothing —
 * and neither says which part of the request was too big. This measures the
 * request the way it is about to be sent, split into the parts the operator can
 * act on, so a failure can name the MCP server responsible.
 *
 * Sizes are `Token.estimate` (~4 characters per token), not a tokenizer — every
 * provider tokenizes differently, and the point is the proportions. A total the
 * provider actually reported replaces the estimate (`measured`).
 */
export namespace RequestFootprint {
  export interface Group {
    tools: number
    tokens: number
  }

  export interface McpServer extends Group {
    server: string
  }

  export interface Info {
    /** `providerID/modelID`, for the message. */
    model: string
    total: number
    system: number
    /** Text of the conversation; binary parts are not in it (see `media`). */
    conversation: number
    /** Binary parts (images, PDFs, files) left out of `conversation` — the estimate cannot price them. */
    media: number
    builtinTools: Group
    /** Largest first. */
    mcp: McpServer[]
    /** 0 when the model's limits are unknown. */
    inputLimit: number
    contextLimit: number
    /** `total` came from the provider's usage report rather than the estimate. */
    measured: boolean
  }

  /** Why the footprint is being reported — each reads as a different first sentence. */
  export type Reason =
    | { type: "too-large" }
    | { type: "empty"; finish?: string }
    | { type: "no-room"; threshold: number }
    /** Refused as too large — by the provider, or by the pre-flight check — after a compaction. */
    | { type: "rejected"; times: number }
    | { type: "compaction-loop"; times: number }

  /**
   * The largest request the model accepts: its input limit, or — when the
   * catalog only knows the window — the window minus the output it must leave
   * room for. Same formula `isOverflow` uses.
   */
  export function inputLimit(model: Provider.Model) {
    if (model.limit.input) return model.limit.input
    if (!model.limit.context) return 0
    return Math.max(0, model.limit.context - ProviderTransform.maxOutputTokens(model))
  }

  export async function measure(input: {
    model: Provider.Model
    system: string[]
    messages: ModelMessage[]
    tools: Record<string, Tool>
    /** The MCP server a tool came from; undefined for a built-in tool. */
    serverOf: (tool: Tool) => string | undefined
  }): Promise<Info> {
    const builtinTools: Group = { tools: 0, tokens: 0 }
    const servers = new Map<string, Group>()
    for (const [name, tool] of Object.entries(input.tools)) {
      const tokens = Token.estimate(await serializeTool(name, tool))
      const server = input.serverOf(tool)
      const group = server === undefined ? builtinTools : (servers.get(server) ?? { tools: 0, tokens: 0 })
      group.tools++
      group.tokens += tokens
      if (server !== undefined) servers.set(server, group)
    }
    const mcp = [...servers].map(([server, group]) => ({ server, ...group })).sort((a, b) => b.tokens - a.tokens)
    const system = Token.estimate(input.system.join("\n"))
    const serialized = serializeMessages(input.messages)
    const conversation = Token.estimate(serialized.text)
    return {
      model: `${input.model.providerID}/${input.model.id}`,
      total: system + conversation + builtinTools.tokens + mcp.reduce((sum, server) => sum + server.tokens, 0),
      system,
      conversation,
      media: serialized.media,
      builtinTools,
      mcp,
      inputLimit: inputLimit(input.model),
      contextLimit: input.model.limit.context,
      measured: false,
    }
  }

  /**
   * What to do with a request before it is sent. `threshold` is where
   * auto-compaction kicks in, undefined when it is off.
   *
   * - `send` — it fits, or the model's limits are unknown.
   * - `compact` — it does not fit, but the excess is in the conversation, which
   *   compaction can shrink: the same remedy a provider rejection leads to,
   *   without spending the request.
   * - `too-large` — it does not fit and compaction cannot help: the fixed part
   *   alone reaches the threshold, or auto-compaction is off.
   */
  export function preflight(info: Info, threshold: number | undefined): "send" | "compact" | "too-large" {
    if (info.inputLimit <= 0 || info.total <= info.inputLimit) return "send"
    if (threshold === undefined || fixed(info) >= threshold) return "too-large"
    return "compact"
  }

  /** Replace the estimated total with the input the provider reported for this request. */
  export function withMeasuredInput(
    info: Info,
    tokens: { input: number; cache: { read: number; write: number } },
  ): Info {
    const total = tokens.input + tokens.cache.read + tokens.cache.write
    if (total <= 0) return info
    return { ...info, total, measured: true }
  }

  /**
   * What every request carries no matter how short the conversation is — the
   * part compaction cannot touch — and whether that figure is a measurement.
   *
   * With a measured total the conversation estimate is subtracted from it, so
   * a tool block the estimate undershoots still counts at its real size. Not
   * when the conversation holds binary parts: their tokens are in the measured
   * total but not in the estimate, so the difference would book every
   * screenshot as fixed overhead — which compaction (it strips media) can in
   * fact remove.
   */
  export function fixedOverhead(info: Info): { tokens: number; measured: boolean } {
    const estimated = info.system + info.builtinTools.tokens + info.mcp.reduce((sum, server) => sum + server.tokens, 0)
    if (!info.measured || info.media > 0) return { tokens: estimated, measured: false }
    const derived = info.total - info.conversation
    return derived > estimated ? { tokens: derived, measured: true } : { tokens: estimated, measured: false }
  }

  export function fixed(info: Info) {
    return fixedOverhead(info).tokens
  }

  /** One English paragraph: what happened, the numbers behind it, what to do. */
  export function format(info: Info, reason: Reason) {
    const size = `${info.measured ? "" : "≈"}${n(info.total)} input tokens`
    const limit = info.inputLimit
      ? `the model accepts ${n(info.inputLimit)} input tokens (context window ${n(info.contextLimit)})`
      : `the model's input limit is unknown`
    const sentences: string[] = []
    switch (reason.type) {
      case "too-large":
        sentences.push(`Request too large for ${info.model}: ${size}, but ${limit}.`)
        break
      case "empty":
        sentences.push(
          `${info.model} returned no content (finish: ${reason.finish ?? "none"}). The request was ${size}; ${limit}.`,
        )
        break
      case "no-room": {
        const overhead = fixedOverhead(info)
        sentences.push(
          `Compacting the conversation cannot make room on ${info.model}: the system prompt and tools alone take ` +
            `${overhead.measured ? "" : "≈"}${n(overhead.tokens)} input tokens, above the ` +
            `${n(reason.threshold)}-token compaction threshold; ${limit}.`,
        )
        break
      }
      case "rejected":
        sentences.push(
          `The request to ${info.model} was too large ${reason.times} times in a row, and compacting the ` +
            `conversation did not help. The request was ${size}; ${limit}.`,
        )
        break
      case "compaction-loop":
        sentences.push(
          `Stopped after ${reason.times} compactions in a row on ${info.model}: the compacted request still did not ` +
            `fit. The request was ${size}; ${limit}.`,
        )
        break
    }
    if (info.mcp.length > 0) {
      const total = info.mcp.reduce((sum, server) => sum + server.tokens, 0)
      const servers = info.mcp.map((server) => `${server.server} ≈${n(server.tokens)} (${tools(server.tools)})`)
      sentences.push(`MCP tools ≈${n(total)}: ${servers.join(", ")}.`)
    }
    sentences.push(
      `Built-in tools ≈${n(info.builtinTools.tokens)} (${tools(info.builtinTools.tools)}), ` +
        `system prompt ≈${n(info.system)}, conversation ≈${n(info.conversation)}.`,
    )
    if (reason.type === "empty") sentences.push("Large tool sets can make a model fail even below its input limit.")
    sentences.push(
      info.mcp.length > 0
        ? "Disable the largest MCP server or pick a model with a larger input limit."
        : "Pick a model with a larger input limit.",
    )
    return sentences.join(" ")
  }

  function n(value: number) {
    return value.toLocaleString("en-US")
  }

  function tools(count: number) {
    return `${count} tool${count === 1 ? "" : "s"}`
  }

  /** The tool as the provider receives it: name, description, JSON schema. */
  async function serializeTool(name: string, tool: Tool) {
    const parameters = tool.inputSchema ? await asSchema(tool.inputSchema).jsonSchema : {}
    return JSON.stringify({ name, description: tool.description ?? "", parameters })
  }

  // Parts whose payload is binary (images, PDFs, other files). Their base64
  // would count as hundreds of thousands of "tokens" of text, while providers
  // price them on their own terms — so they are left out rather than guessed.
  const MEDIA_PARTS = new Set(["image", "file", "media", "image-data", "file-data"])

  function serializeMessages(messages: ModelMessage[]) {
    let media = 0
    const text = JSON.stringify(messages, (_key, value: unknown) => {
      if (value instanceof Uint8Array || value instanceof ArrayBuffer) return undefined
      if (typeof value === "object" && value !== null && "type" in value && MEDIA_PARTS.has(String(value.type))) {
        media++
        return { type: value.type }
      }
      return value
    })
    return { text, media }
  }
}
