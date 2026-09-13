import { describe, expect, test } from "bun:test"
import { isCodexOauthModel, codexContextLimit } from "../../src/plugin/codex"

function model(id: string, options?: Record<string, unknown>) {
  return { api: { id }, options }
}

describe("plugin.codex model rule", () => {
  describe("isCodexOauthModel", () => {
    // The whole point of the version rule: models that did not exist when this
    // code was written must be admitted without touching it. A frozen list is
    // what caused "Model not found: openai/gpt-5.6-luna" in the first place.
    test("admits models newer than gpt-5.4 that this code has never heard of", () => {
      expect(isCodexOauthModel(model("gpt-5.6-luna"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-5.6-terra"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-5.6-sol"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-6-astra"))).toBe(true)
      // Hypothetical future releases — no code change may be needed for these.
      expect(isCodexOauthModel(model("gpt-5.9-whatever"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-7"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-12.3-newthing"))).toBe(true)
    })

    test("admits the explicitly listed current models", () => {
      expect(isCodexOauthModel(model("gpt-5.5"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-5.3-codex-spark"))).toBe(true)
      // Retired from Codex on 2026-08-31 but still listed upstream: the backend
      // is the authority on those, we do not second-guess it here.
      expect(isCodexOauthModel(model("gpt-5.4"))).toBe(true)
      expect(isCodexOauthModel(model("gpt-5.4-mini"))).toBe(true)
    })

    test("rejects generations at or below gpt-5.4 that are not listed", () => {
      expect(isCodexOauthModel(model("gpt-5.2"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-5.3-codex"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-5.1-codex-max"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-5"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-4.1-mini"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-5.4-nano"))).toBe(false)
    })

    test("rejects the explicitly disallowed models", () => {
      expect(isCodexOauthModel(model("gpt-5.5-pro"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-5.6"))).toBe(false)
    })

    test("rejects pro reasoning variants — billed outside the subscription", () => {
      // Caught by the ID suffix today...
      expect(isCodexOauthModel(model("gpt-5.6-pro"))).toBe(false)
      expect(isCodexOauthModel(model("gpt-9.9-pro"))).toBe(false)
      // ...and by the option once a newer upstream starts populating it.
      expect(isCodexOauthModel(model("gpt-5.6-luna", { reasoningMode: "pro" }))).toBe(false)
    })

    test("rejects non-gpt model ids", () => {
      expect(isCodexOauthModel(model("o3-mini"))).toBe(false)
      expect(isCodexOauthModel(model("text-embedding-3-large"))).toBe(false)
      expect(isCodexOauthModel(model("chatgpt-4o-latest"))).toBe(false)
    })

    test("compares versions numerically, not as strings", () => {
      // "gpt-5.10" sorts before "gpt-5.4" as a string but is a later generation.
      expect(isCodexOauthModel(model("gpt-5.10-something"))).toBe(true)
    })
  })

  describe("codexContextLimit", () => {
    const apiLimit = { context: 1_050_000, input: 922_000, output: 128_000 }

    test("caps 5.5/5.6 models to the window Codex actually grants", () => {
      expect(codexContextLimit({ id: "gpt-5.6-luna", limit: apiLimit })).toEqual({
        context: 400_000,
        input: 272_000,
        output: 128_000,
      })
      expect(codexContextLimit({ id: "gpt-5.5", limit: apiLimit }).context).toBe(400_000)
    })

    test("leaves other models' limits untouched", () => {
      const other = { context: 200_000, input: 190_000, output: 32_000 }
      expect(codexContextLimit({ id: "gpt-5.4-mini", limit: other })).toBe(other)
      expect(codexContextLimit({ id: "gpt-6-astra", limit: other })).toBe(other)
    })
  })
})
