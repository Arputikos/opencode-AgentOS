import { describe, expect, test } from "bun:test"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionID, MessageID } from "../../src/session/schema"

describe("structured-output.OutputFormat", () => {
  test("parses text format", () => {
    const result = MessageV2.Format.safeParse({ type: "text" })
    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data.type).toBe("text")
    }
  })

  test("parses json_schema format with defaults", () => {
    const result = MessageV2.Format.safeParse({
      type: "json_schema",
      schema: { type: "object", properties: { name: { type: "string" } } },
    })
    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data.type).toBe("json_schema")
      if (result.data.type === "json_schema") {
        expect(result.data.retryCount).toBe(2) // default value
      }
    }
  })

  test("parses json_schema format with custom retryCount", () => {
    const result = MessageV2.Format.safeParse({
      type: "json_schema",
      schema: { type: "object" },
      retryCount: 5,
    })
    expect(result.success).toBe(true)
    if (result.success && result.data.type === "json_schema") {
      expect(result.data.retryCount).toBe(5)
    }
  })

  test("rejects invalid type", () => {
    const result = MessageV2.Format.safeParse({ type: "invalid" })
    expect(result.success).toBe(false)
  })

  test("rejects json_schema without schema", () => {
    const result = MessageV2.Format.safeParse({ type: "json_schema" })
    expect(result.success).toBe(false)
  })

  test("rejects negative retryCount", () => {
    const result = MessageV2.Format.safeParse({
      type: "json_schema",
      schema: { type: "object" },
      retryCount: -1,
    })
    expect(result.success).toBe(false)
  })
})

describe("structured-output.StructuredOutputError", () => {
  test("creates error with message and retries", () => {
    const error = new MessageV2.StructuredOutputError({
      message: "Failed to validate",
      retries: 3,
    })

    expect(error.name).toBe("StructuredOutputError")
    expect(error.data.message).toBe("Failed to validate")
    expect(error.data.retries).toBe(3)
  })

  test("converts to object correctly", () => {
    const error = new MessageV2.StructuredOutputError({
      message: "Test error",
      retries: 2,
    })

    const obj = error.toObject()
    expect(obj.name).toBe("StructuredOutputError")
    expect(obj.data.message).toBe("Test error")
    expect(obj.data.retries).toBe(2)
  })

  test("isInstance correctly identifies error", () => {
    const error = new MessageV2.StructuredOutputError({
      message: "Test",
      retries: 1,
    })

    expect(MessageV2.StructuredOutputError.isInstance(error)).toBe(true)
    expect(MessageV2.StructuredOutputError.isInstance({ name: "other" })).toBe(false)
  })
})

describe("structured-output.UserMessage", () => {
  test("user message accepts outputFormat", () => {
    const result = MessageV2.User.safeParse({
      id: MessageID.ascending(),
      sessionID: SessionID.descending(),
      role: "user",
      time: { created: Date.now() },
      agent: "default",
      model: { providerID: "anthropic", modelID: "claude-3" },
      outputFormat: {
        type: "json_schema",
        schema: { type: "object" },
      },
    })
    expect(result.success).toBe(true)
  })

  test("user message works without outputFormat (optional)", () => {
    const result = MessageV2.User.safeParse({
      id: MessageID.ascending(),
      sessionID: SessionID.descending(),
      role: "user",
      time: { created: Date.now() },
      agent: "default",
      model: { providerID: "anthropic", modelID: "claude-3" },
    })
    expect(result.success).toBe(true)
  })
})

describe("structured-output.AssistantMessage", () => {
  const baseAssistantMessage = {
    id: MessageID.ascending(),
    sessionID: SessionID.descending(),
    role: "assistant" as const,
    parentID: MessageID.ascending(),
    modelID: "claude-3",
    providerID: "anthropic",
    mode: "default",
    agent: "default",
    path: { cwd: "/test", root: "/test" },
    cost: 0.001,
    tokens: { input: 100, output: 50, reasoning: 0, cache: { read: 0, write: 0 } },
    time: { created: Date.now() },
  }

  test("assistant message accepts structured", () => {
    const result = MessageV2.Assistant.safeParse({
      ...baseAssistantMessage,
      structured: { company: "Anthropic", founded: 2021 },
    })
    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data.structured).toEqual({ company: "Anthropic", founded: 2021 })
    }
  })

  test("assistant message works without structured_output (optional)", () => {
    const result = MessageV2.Assistant.safeParse(baseAssistantMessage)
    expect(result.success).toBe(true)
  })
})

describe("structured-output.createStructuredOutputTool", () => {
  // Patched (agentos): the tool is a PASS-THROUGH. Validation, the corrective
  // retry budget and the failure all live in the Agent OS orchestrator, so the
  // fork is only responsible for two things — showing the model the caller's
  // real JSON Schema, and relaying the orchestrator's verdict back. The
  // validation behaviour these tests used to assert is covered by the
  // orchestrator's own suite (`core/__tests__/structured-output.test.ts`).
  const accept = async () => ({ accepted: true, text: "Structured output captured successfully." })
  const reject = async () => ({ accepted: false, text: "did NOT match the required schema" })
  const callOpts = { toolCallId: "test-call-id", messages: [], abortSignal: undefined as any }

  test("creates tool with correct id", () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", properties: { name: { type: "string" } } },
      submit: accept,
      onAccepted: () => {},
    })

    // AI SDK tool type doesn't expose id, but we set it internally
    expect((tool as any).id).toBe("StructuredOutput")
  })

  test("creates tool with description", () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object" },
      submit: accept,
      onAccepted: () => {},
    })

    expect(tool.description).toContain("structured format")
  })

  test("hands the model the caller's schema verbatim as inputSchema", () => {
    const schema = {
      type: "object",
      properties: {
        company: { type: "string" },
        founded: { type: "number" },
        tags: { type: "array", items: { type: "string" } },
        user: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
      },
      required: ["company"],
    }

    const tool = SessionPrompt.createStructuredOutputTool({
      schema,
      submit: accept,
      onAccepted: () => {},
    })

    // This is the one thing only the fork can do: the pass-through `jsonSchema()`
    // bypasses the zod-only tool registry, so oneOf/$ref/nested shapes survive.
    const inputSchema = tool.inputSchema as any
    expect(inputSchema.jsonSchema?.properties?.company).toBeDefined()
    expect(inputSchema.jsonSchema?.properties?.founded?.type).toBe("number")
    expect(inputSchema.jsonSchema?.properties?.tags?.items?.type).toBe("string")
    expect(inputSchema.jsonSchema?.properties?.user?.required).toContain("name")
    expect(inputSchema.jsonSchema?.required).toContain("company")
  })

  test("strips $schema property from inputSchema", () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: {
        $schema: "http://json-schema.org/draft-07/schema#",
        type: "object",
        properties: { name: { type: "string" } },
      },
      submit: accept,
      onAccepted: () => {},
    })

    const inputSchema = tool.inputSchema as any
    expect(inputSchema.jsonSchema?.$schema).toBeUndefined()
  })

  test("execute submits the model's argument verbatim and reports acceptance", async () => {
    let submitted: unknown
    let accepted = false

    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", properties: { name: { type: "string" } } },
      submit: async (args) => {
        submitted = args
        return { accepted: true, text: "Structured output captured successfully." }
      },
      onAccepted: () => {
        accepted = true
      },
    })

    const testArgs = { name: "Test Company", nested: { deep: [1, 2, 3] } }
    const result = await tool.execute!(testArgs, callOpts)

    expect(submitted).toEqual(testArgs)
    expect(accepted).toBe(true)
    expect(result.output).toBe("Structured output captured successfully.")
    expect(result.metadata.valid).toBe(true)
  })

  test("a rejected submission returns the orchestrator's text and does NOT end the turn", async () => {
    let accepted = false

    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", required: ["sentiment"] },
      submit: reject,
      onAccepted: () => {
        accepted = true
      },
    })

    const result = await tool.execute!({ sentiment: "banana" }, callOpts)

    // `onAccepted` is what breaks the run loop — a rejection must leave the
    // model free to correct itself on the next step.
    expect(accepted).toBe(false)
    expect(result.metadata.valid).toBe(false)
    expect(result.output).toContain("did NOT match the required schema")
  })

  test("toModelOutput returns text value", async () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object" },
      submit: accept,
      onAccepted: () => {},
    })

    expect(tool.toModelOutput).toBeDefined()
    const modelOutput = await Promise.resolve(
      tool.toModelOutput!({
        toolCallId: "test-call-id",
        input: {},
        output: {
          output: "Test output",
        },
      }),
    )

    expect(modelOutput.type).toBe("text")
    if (modelOutput.type !== "text") throw new Error("expected text model output")
    expect(modelOutput.value).toBe("Test output")
  })
})
