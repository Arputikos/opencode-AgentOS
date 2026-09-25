import { describe, expect, test } from "bun:test"
import { dynamicTool, jsonSchema, type ModelMessage, type Tool } from "ai"
import type { Provider } from "../../src/provider/provider"
import { RequestFootprint } from "../../src/session/request-footprint"

// Pure — no instance, no temp directory.

function model(limit: { context: number; input?: number; output: number }): Provider.Model {
  return { id: "gpt-5.6-terra", providerID: "openai", limit } as Provider.Model
}

// gpt-5.6-* on a ChatGPT subscription, as the Codex plugin clamps it.
const subscription = model({ context: 400_000, input: 272_000, output: 128_000 })

/** A tool whose serialized definition is roughly `chars` characters. */
function toolOf(chars: number): Tool {
  return dynamicTool({
    description: "x".repeat(chars),
    inputSchema: jsonSchema({ type: "object", properties: {} }),
    execute: async () => ({}),
  })
}

async function measure(input: {
  tools: Record<string, { tool: Tool; server?: string }>
  system?: string[]
  messages?: ModelMessage[]
  model?: Provider.Model
}) {
  const servers = new Map(Object.values(input.tools).map((entry) => [entry.tool, entry.server]))
  return RequestFootprint.measure({
    model: input.model ?? subscription,
    system: input.system ?? [],
    messages: input.messages ?? [],
    tools: Object.fromEntries(Object.entries(input.tools).map(([name, entry]) => [name, entry.tool])),
    serverOf: (tool) => servers.get(tool),
  })
}

describe("RequestFootprint", () => {
  describe("inputLimit", () => {
    test("uses the input limit when the catalog has one", () => {
      expect(RequestFootprint.inputLimit(subscription)).toBe(272_000)
    })

    test("otherwise the window minus the output reserve — the formula isOverflow uses", () => {
      // maxOutputTokens caps the reserve at 32k.
      expect(RequestFootprint.inputLimit(model({ context: 1_050_000, output: 128_000 }))).toBe(1_018_000)
    })

    test("0 when the model's limits are unknown", () => {
      expect(RequestFootprint.inputLimit(model({ context: 0, output: 0 }))).toBe(0)
    })
  })

  describe("measure", () => {
    test("groups tools by the server that registered them, not by the tool key", async () => {
      // Server names contain `_` and `-` freely, so the `<server>_<tool>` key
      // cannot be split back — the owner comes from `serverOf`.
      const fp = await measure({
        tools: {
          my_eleven_labs_text_to_speech: { tool: toolOf(4_000), server: "my_eleven-labs" },
          my_eleven_labs_voices: { tool: toolOf(4_000), server: "my_eleven-labs" },
          senses_look: { tool: toolOf(400), server: "senses" },
          bash: { tool: toolOf(800) },
          read: { tool: toolOf(800) },
        },
      })
      expect(fp.mcp.map((server) => [server.server, server.tools])).toEqual([
        ["my_eleven-labs", 2],
        ["senses", 1],
      ])
      expect(fp.builtinTools.tools).toBe(2)
      expect(fp.mcp[0].tokens).toBeGreaterThan(2_000)
      expect(fp.total).toBe(fp.system + fp.conversation + fp.builtinTools.tokens + fp.mcp[0].tokens + fp.mcp[1].tokens)
    })

    test("sorts MCP servers largest first", async () => {
      const fp = await measure({
        tools: {
          a_small: { tool: toolOf(100), server: "small" },
          b_big: { tool: toolOf(10_000), server: "big" },
          c_mid: { tool: toolOf(1_000), server: "mid" },
        },
      })
      expect(fp.mcp.map((server) => server.server)).toEqual(["big", "mid", "small"])
    })

    test("counts the tool schema, not only its description", async () => {
      const bare = await measure({ tools: { t: { tool: toolOf(0), server: "s" } } })
      const schema = dynamicTool({
        description: "",
        inputSchema: jsonSchema({
          type: "object",
          properties: Object.fromEntries(
            Array.from({ length: 50 }, (_, i) => [`field${i}`, { type: "string", description: "y".repeat(80) }]),
          ),
        }),
        execute: async () => ({}),
      })
      const heavy = await measure({ tools: { t: { tool: schema, server: "s" } } })
      expect(heavy.mcp[0].tokens - bare.mcp[0].tokens).toBeGreaterThan(1_000)
    })

    test("leaves binary attachments out of the conversation estimate", async () => {
      // 400 KB of base64 would otherwise read as ~100k "tokens" of text.
      const image = "A".repeat(400_000)
      const fp = await measure({
        tools: {},
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "what is on this picture?" },
              { type: "image", image, mediaType: "image/png" },
            ],
          },
        ],
      })
      expect(fp.conversation).toBeGreaterThan(0)
      expect(fp.conversation).toBeLessThan(100)
    })
  })

  describe("preflight", () => {
    const threshold = 252_000

    test("sends a request that fits", async () => {
      const fp = await measure({ tools: { t: { tool: toolOf(400_000), server: "s" } } })
      expect(fp.total).toBeLessThan(272_000)
      expect(RequestFootprint.preflight(fp, threshold)).toBe("send")
    })

    test("refuses a request whose tools alone do not fit — compaction cannot help", async () => {
      const fp = await measure({ tools: { t: { tool: toolOf(1_300_000), server: "elevenlabs" } } })
      expect(fp.total).toBeGreaterThan(272_000)
      expect(RequestFootprint.preflight(fp, threshold)).toBe("too-large")
    })

    test("compacts when the excess is in the conversation", async () => {
      const fp = await measure({
        tools: { t: { tool: toolOf(40_000), server: "s" } },
        messages: [{ role: "user", content: "z".repeat(1_200_000) }],
      })
      expect(fp.total).toBeGreaterThan(272_000)
      expect(RequestFootprint.preflight(fp, threshold)).toBe("compact")
    })

    test("refuses instead of compacting when auto-compaction is off", async () => {
      const fp = await measure({ tools: {}, messages: [{ role: "user", content: "z".repeat(1_200_000) }] })
      expect(RequestFootprint.preflight(fp, undefined)).toBe("too-large")
    })

    test("sends anything when the model's limits are unknown", async () => {
      const fp = await measure({
        model: model({ context: 0, output: 0 }),
        tools: { t: { tool: toolOf(1_300_000), server: "s" } },
      })
      expect(RequestFootprint.preflight(fp, undefined)).toBe("send")
    })
  })

  describe("withMeasuredInput / fixed", () => {
    test("a reported total replaces the estimate, cache included", async () => {
      const fp = await measure({ tools: { t: { tool: toolOf(4_000), server: "s" } } })
      const measured = RequestFootprint.withMeasuredInput(fp, { input: 570, cache: { read: 325_120, write: 0 } })
      expect(measured.measured).toBe(true)
      expect(measured.total).toBe(325_690)
    })

    test("zero usage is not a measurement", async () => {
      // The luna failure reports no usage at all.
      const fp = await measure({ tools: { t: { tool: toolOf(4_000), server: "s" } } })
      expect(RequestFootprint.withMeasuredInput(fp, { input: 0, cache: { read: 0, write: 0 } })).toBe(fp)
    })

    test("a measured total lifts a fixed overhead the estimate undershot", async () => {
      const fp = await measure({ tools: { t: { tool: toolOf(40_000), server: "s" } } })
      expect(RequestFootprint.fixed(fp)).toBeLessThan(20_000)
      const measured = RequestFootprint.withMeasuredInput(fp, { input: 300_000, cache: { read: 0, write: 0 } })
      expect(RequestFootprint.fixed(measured)).toBe(300_000 - fp.conversation)
    })

    test("never books binary parts as fixed overhead", async () => {
      // Screenshots are in the measured total but not in the estimate, and
      // compaction strips them — so the difference is not "fixed".
      const fp = await measure({
        tools: { t: { tool: toolOf(40_000), server: "s" } },
        messages: [{ role: "user", content: [{ type: "image", image: "AAAA", mediaType: "image/png" }] }],
      })
      expect(fp.media).toBe(1)
      const measured = RequestFootprint.withMeasuredInput(fp, { input: 128_000, cache: { read: 0, write: 0 } })
      expect(RequestFootprint.fixedOverhead(measured)).toEqual({ tokens: RequestFootprint.fixed(fp), measured: false })
    })
  })

  describe("format", () => {
    const fp: RequestFootprint.Info = {
      model: "openai/gpt-5.6-terra",
      total: 325_500,
      system: 4_000,
      conversation: 3_000,
      builtinTools: { tools: 40, tokens: 9_000 },
      mcp: [
        { server: "elevenlabs", tools: 120, tokens: 302_100 },
        { server: "senses", tools: 1, tokens: 7_400 },
      ],
      inputLimit: 272_000,
      contextLimit: 400_000,
      measured: false,
      media: 0,
    }

    test("too large: request size, input limit, window and the per-server breakdown", () => {
      expect(RequestFootprint.format(fp, { type: "too-large" })).toBe(
        "Request too large for openai/gpt-5.6-terra: ≈325,500 input tokens, but the model accepts 272,000 input " +
          "tokens (context window 400,000). MCP tools ≈309,500: elevenlabs ≈302,100 (120 tools), senses ≈7,400 " +
          "(1 tool). Built-in tools ≈9,000 (40 tools), system prompt ≈4,000, conversation ≈3,000. Disable the " +
          "largest MCP server or pick a model with a larger input limit.",
      )
    })

    test("a measured total drops the ≈", () => {
      const text = RequestFootprint.format({ ...fp, measured: true }, { type: "compaction-loop", times: 2 })
      expect(text).toContain("The request was 325,500 input tokens;")
      expect(text).toStartWith("Stopped after 2 compactions in a row on openai/gpt-5.6-terra")
    })

    test("empty turn: the same numbers plus why a model can fail below its limit", () => {
      const text = RequestFootprint.format(
        { ...fp, model: "openai/gpt-6-luna", inputLimit: 922_000, contextLimit: 1_050_000 },
        { type: "empty", finish: "length" },
      )
      expect(text).toStartWith(
        "openai/gpt-6-luna returned no content (finish: length). The request was ≈325,500 input tokens; the model " +
          "accepts 922,000 input tokens (context window 1,050,000).",
      )
      expect(text).toContain("elevenlabs ≈302,100 (120 tools)")
      expect(text).toContain("Large tool sets can make a model fail even below its input limit.")
    })

    test("no room: names the fixed overhead and the threshold", () => {
      expect(RequestFootprint.format(fp, { type: "no-room", threshold: 252_000 })).toStartWith(
        "Compacting the conversation cannot make room on openai/gpt-5.6-terra: the system prompt and tools alone " +
          "take ≈322,500 input tokens, above the 252,000-token compaction threshold; the model accepts 272,000 input " +
          "tokens (context window 400,000).",
      )
    })

    test("no room: a measured overhead is stated without ≈", () => {
      const text = RequestFootprint.format(
        { ...fp, measured: true, total: 330_000 },
        { type: "no-room", threshold: 252_000 },
      )
      expect(text).toContain("tools alone take 327,000 input tokens")
    })

    test("without MCP servers there is no MCP sentence and no MCP advice", () => {
      const text = RequestFootprint.format({ ...fp, mcp: [] }, { type: "too-large" })
      expect(text).not.toContain("MCP")
      expect(text).toEndWith("Pick a model with a larger input limit.")
    })

    test("unknown limits are said to be unknown, not zero", () => {
      expect(RequestFootprint.format({ ...fp, inputLimit: 0, contextLimit: 0 }, { type: "empty" })).toContain(
        "the model's input limit is unknown",
      )
    })
  })
})
