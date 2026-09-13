import { describe, expect, test } from "bun:test"
import { CodexAuthPlugin } from "../../src/plugin/codex"

/**
 * Exercises the hook the way OpenCode's provider state does, one level below
 * the HTTP contract test: `CodexAuthPlugin(...).provider.models(provider, ctx)`
 * is exactly what `provider.ts` calls to build the catalog a session sees.
 *
 * Proves the whole ticket-1 symptom without a live server:
 *   - under OAuth, gpt-5.6-luna SURVIVES (it used to be deleted → "Model not
 *     found: openai/gpt-5.6-luna" and the request never left the machine),
 *   - stale entries like gpt-5.2 are gone,
 *   - the Codex context limit replaces the public-API one,
 *   - with an API key the catalog is returned untouched.
 */

// Shape mirrors what models.dev yields after provider.ts normalises it: the
// fields the hook reads (`id`, `api.id`, `cost`, `limit`, `options`) plus a few
// carried through untouched, so we can assert nothing else is mangled.
function model(id: string, limit?: { context: number; input?: number; output: number }) {
  return {
    id,
    api: { id },
    name: id,
    cost: { input: 1.25, output: 10, cache: { read: 0.125, write: 0.5 } },
    limit: limit ?? { context: 1_050_000, input: 922_000, output: 128_000 },
    options: {},
    release_date: "2026-07-09",
  }
}

const CATALOG = {
  // Currently usable on a ChatGPT subscription.
  "gpt-5.6-luna": model("gpt-5.6-luna"),
  "gpt-5.6-terra": model("gpt-5.6-terra"),
  "gpt-5.6-sol": model("gpt-5.6-sol"),
  "gpt-6-astra": model("gpt-6-astra", { context: 400_000, output: 128_000 }),
  "gpt-5.5": model("gpt-5.5"),
  "gpt-5.3-codex-spark": model("gpt-5.3-codex-spark", { context: 272_000, output: 128_000 }),
  // Retired from Codex 2026-08-31 but still listed upstream — the backend, not
  // us, refuses these.
  "gpt-5.4": model("gpt-5.4", { context: 400_000, output: 128_000 }),
  "gpt-5.4-mini": model("gpt-5.4-mini", { context: 400_000, output: 128_000 }),
  // Deprecated / never on a subscription.
  "gpt-5.2": model("gpt-5.2", { context: 400_000, output: 128_000 }),
  "gpt-5.1-codex-max": model("gpt-5.1-codex-max", { context: 272_000, output: 100_000 }),
  "gpt-5.5-pro": model("gpt-5.5-pro"),
  "gpt-5.4-pro": model("gpt-5.4-pro"),
  "gpt-4.1-mini": model("gpt-4.1-mini", { context: 1_047_576, output: 32_768 }),
}

async function catalogFor(authType: "oauth" | "api") {
  const hooks = await CodexAuthPlugin({ client: {} } as never)
  const models = hooks.provider?.models
  if (!models) throw new Error("CodexAuthPlugin no longer exposes a provider.models hook")
  const provider = { id: "openai", models: structuredClone(CATALOG) }
  return models(provider as never, { auth: { type: authType } } as never)
}

describe("plugin.codex provider.models hook", () => {
  test("exposes the hook under the openai provider id", async () => {
    const hooks = await CodexAuthPlugin({ client: {} } as never)
    expect(hooks.provider?.id).toBe("openai")
    expect(typeof hooks.provider?.models).toBe("function")
  })

  describe("with an OAuth (ChatGPT subscription) credential", () => {
    test("keeps the models a subscription can actually call today", async () => {
      const out = await catalogFor("oauth")
      // The headline regression: this exact id produced "Model not found:
      // openai/gpt-5.6-luna" in production.
      expect(Object.keys(out)).toContain("gpt-5.6-luna")
      expect(Object.keys(out)).toEqual(
        expect.arrayContaining(["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra", "gpt-5.5"]),
      )
    })

    test("drops deprecated generations and pro variants", async () => {
      const ids = Object.keys(await catalogFor("oauth"))
      for (const gone of ["gpt-5.2", "gpt-5.1-codex-max", "gpt-4.1-mini", "gpt-5.5-pro", "gpt-5.4-pro"]) {
        expect(ids).not.toContain(gone)
      }
    })

    test("replaces the public-API context limit with the one Codex grants", async () => {
      const out = await catalogFor("oauth")
      // models.dev says 1_050_000 for luna — that is the API limit, not Codex's.
      expect(CATALOG["gpt-5.6-luna"].limit.context).toBe(1_050_000)
      expect(out["gpt-5.6-luna"]!.limit).toEqual({ context: 400_000, input: 272_000, output: 128_000 })
      expect(out["gpt-5.5"]!.limit.context).toBe(400_000)
    })

    test("leaves the limit alone for models outside the 5.5/5.6 override", async () => {
      const out = await catalogFor("oauth")
      expect(out["gpt-6-astra"]!.limit).toEqual(CATALOG["gpt-6-astra"].limit)
    })

    test("zeroes cost — usage is covered by the subscription, not billed per token", async () => {
      const out = await catalogFor("oauth")
      for (const m of Object.values(out)) {
        expect(m.cost).toEqual({ input: 0, output: 0, cache: { read: 0, write: 0 } })
      }
    })

    test("does not mutate the catalog it was handed", async () => {
      // The old implementation deleted from `provider.models` in place. Anything
      // else holding that object saw a half-empty catalog.
      const hooks = await CodexAuthPlugin({ client: {} } as never)
      const provider = { id: "openai", models: structuredClone(CATALOG) }
      const before = Object.keys(provider.models).length
      await hooks.provider!.models!(provider as never, { auth: { type: "oauth" } } as never)
      expect(Object.keys(provider.models)).toHaveLength(before)
      expect(provider.models["gpt-5.2"]).toBeDefined()
    })
  })

  describe("with an API key", () => {
    test("returns the catalog untouched — no filtering, no cost/limit rewrite", async () => {
      const out = await catalogFor("api")
      expect(Object.keys(out).sort()).toEqual(Object.keys(CATALOG).sort())
      expect(out["gpt-5.6-luna"]!.limit.context).toBe(1_050_000)
      expect(out["gpt-5.2"]).toBeDefined();
      expect(out["gpt-5.6-luna"]!.cost.input).toBe(1.25)
    })

    test("returns the catalog untouched when there is no auth at all", async () => {
      const hooks = await CodexAuthPlugin({ client: {} } as never)
      const provider = { id: "openai", models: structuredClone(CATALOG) }
      const out = await hooks.provider!.models!(provider as never, {} as never)
      expect(Object.keys(out).sort()).toEqual(Object.keys(CATALOG).sort())
    })
  })
})
