import { describe, expect, test } from "bun:test"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionPrompt } from "../../src/session/prompt"
import type { MessageV2 } from "../../src/session/message-v2"
import type { Config } from "../../src/config/config"
import type { Provider } from "../../src/provider/provider"
import { compactionThreshold } from "../../src/session/overflow"

// Pure decision logic behind the two overflow patches. Kept separate from
// `compaction.test.ts` because those tests need a real temp directory and this
// file deliberately needs nothing at all.
describe("session overflow guard", () => {
  describe("SessionPrompt.nextOverflowCompaction", () => {
    // Small fixed overhead, threshold 252k (gpt-5.6 on a ChatGPT subscription:
    // input 272k minus the 20k compaction buffer).
    const roomy = { fixed: 20_000, threshold: 252_000 }

    test("allows exactly one compaction — normal recovery", () => {
      expect(SessionPrompt.nextOverflowCompaction({ ...roomy, consecutive: 0 })).toEqual({
        action: "compact",
        consecutive: 1,
      })
    })

    test("gives up when the request built from the compacted conversation needs compacting again", () => {
      // Counts our own pre-emptive compactions too: the production loop never
      // saw a single provider rejection — every step was simply over the
      // threshold again.
      expect(SessionPrompt.nextOverflowCompaction({ ...roomy, consecutive: 1 })).toEqual({
        action: "give-up",
        consecutive: 2,
      })
    })

    test("terminates — the production loop compacted until the operator stopped it", () => {
      // Drive it the way runLoop does when every step asks for a compaction.
      let consecutive = 0
      let compactions = 0
      for (let i = 0; i < 100; i++) {
        const next = SessionPrompt.nextOverflowCompaction({ ...roomy, consecutive })
        consecutive = next.consecutive
        if (next.action !== "compact") break
        compactions++
      }
      expect(compactions).toBe(1)
    })

    test("refuses even the first compaction when the fixed overhead alone reaches the threshold", () => {
      // The ElevenLabs case: ~325k of system prompt + tool schemas against a
      // 252k threshold. Compaction only shrinks the conversation, so it can
      // never get the request under the line.
      expect(SessionPrompt.nextOverflowCompaction({ fixed: 325_000, threshold: 252_000, consecutive: 0 })).toEqual({
        action: "no-room",
        consecutive: 0,
      })
      expect(SessionPrompt.nextOverflowCompaction({ fixed: 252_000, threshold: 252_000, consecutive: 0 }).action).toBe(
        "no-room",
      )
    })

    test("without auto-compaction there is no threshold to compare the overhead with", () => {
      expect(
        SessionPrompt.nextOverflowCompaction({ fixed: 325_000, threshold: undefined, consecutive: 0 }).action,
      ).toBe("compact")
    })
  })

  describe("compactionThreshold", () => {
    const cfg = {} as Config.Info
    const model = (limit: Provider.Model["limit"]) => ({ limit }) as Provider.Model

    test("input limit minus the compaction buffer", () => {
      expect(compactionThreshold({ cfg, model: model({ context: 400_000, input: 272_000, output: 128_000 }) })).toBe(
        252_000,
      )
    })

    test("no threshold when the window is unknown — never a negative one", () => {
      // A model from the config without `limit` has context 0. A threshold of
      // 0 − 32k made every provider rejection end in a bogus "no room".
      expect(compactionThreshold({ cfg, model: model({ context: 0, output: 0 }) })).toBeUndefined()
      expect(
        SessionPrompt.nextOverflowCompaction({
          fixed: 20_000,
          threshold: compactionThreshold({ cfg, model: model({ context: 0, output: 0 }) }),
          consecutive: 0,
        }).action,
      ).toBe("compact")
    })
  })

  describe("SessionPrompt.isEmptyStep", () => {
    const base = { id: "prt_1", sessionID: "ses_1", messageID: "msg_1" } as const
    const markers = [
      { ...base, type: "step-start" },
      { ...base, type: "step-finish", reason: "length" },
    ] as unknown as MessageV2.Part[]

    test("only step markers and finish=length is empty — the luna production case", () => {
      expect(SessionPrompt.isEmptyStep({ parts: markers, finish: "length" })).toBe(true)
    })

    test("finishes that explain nothing count too: other, unknown, none", () => {
      for (const finish of ["other", "unknown", undefined]) {
        expect(SessionPrompt.isEmptyStep({ parts: markers, finish })).toBe(true)
      }
    })

    test("an empty reply with its own explanation is not ours to report", () => {
      // content-filter, or an empty `stop` after tool results — the request's
      // size has nothing to do with it; the orchestrator reports it neutrally.
      for (const finish of ["content-filter", "stop", "tool-calls", "error"]) {
        expect(SessionPrompt.isEmptyStep({ parts: markers, finish })).toBe(false)
      }
    })

    test("an empty text part is still nothing", () => {
      const parts = [{ ...base, type: "text", text: "" }] as unknown as MessageV2.Part[]
      expect(SessionPrompt.isEmptyStep({ parts, finish: "length" })).toBe(true)
    })

    test("text, reasoning or a tool call is something", () => {
      for (const part of [
        { ...base, type: "text", text: "Hi" },
        { ...base, type: "reasoning", text: "…" },
        { ...base, type: "tool", tool: "bash" },
      ]) {
        expect(SessionPrompt.isEmptyStep({ parts: [part] as unknown as MessageV2.Part[], finish: "length" })).toBe(
          false,
        )
      }
    })
  })

  describe("SessionPrompt.shouldDropStaleOverflowCompaction", () => {
    test("drops an overflow compaction queued under a different model", () => {
      // The production case: the verdict was earned by gpt-5.4-mini (400k window),
      // the queued compaction was picked up after a swap to gpt-5.6-luna (1.05M)
      // on a ~501-token conversation.
      expect(SessionPrompt.shouldDropStaleOverflowCompaction({ overflow: true, modelChangedSinceQueued: true })).toBe(
        true,
      )
    })

    test("runs an overflow compaction when the model is still the one that was refused", () => {
      expect(SessionPrompt.shouldDropStaleOverflowCompaction({ overflow: true, modelChangedSinceQueued: false })).toBe(
        false,
      )
    })

    test("never drops an ordinary compaction, model change or not", () => {
      // `overflow: false` is our own token accounting — a statement about the
      // conversation, which no model swap can invalidate.
      for (const modelChangedSinceQueued of [true, false]) {
        expect(SessionPrompt.shouldDropStaleOverflowCompaction({ overflow: false, modelChangedSinceQueued })).toBe(
          false,
        )
      }
    })
  })

  describe("SessionPrompt.isOverflowCompactionCarrier", () => {
    const MODEL_A = { providerID: "openai", modelID: "gpt-5.4-mini" }
    const MODEL_B = { providerID: "openai", modelID: "gpt-5.6-luna" }
    const carrier = (
      model: { providerID: string; modelID: string },
      part: { type: string; overflow?: boolean } = { type: "compaction", overflow: true },
    ) => ({
      info: { role: "user", model },
      parts: [part],
    })

    test("a carrier queued under another model is stale", () => {
      expect(SessionPrompt.isOverflowCompactionCarrier(carrier(MODEL_A), MODEL_B)).toBe(true)
    })

    test("the same model is not stale — the verdict still applies", () => {
      expect(SessionPrompt.isOverflowCompactionCarrier(carrier(MODEL_A), MODEL_A)).toBe(false)
    })

    test("an ordinary compaction carrier is never stale", () => {
      expect(
        SessionPrompt.isOverflowCompactionCarrier(carrier(MODEL_A, { type: "compaction", overflow: false }), MODEL_B),
      ).toBe(false)
      // `overflow` absent entirely — an older part, or our own token accounting.
      expect(SessionPrompt.isOverflowCompactionCarrier(carrier(MODEL_A, { type: "compaction" }), MODEL_B)).toBe(false)
    })

    test("never touches a message that carries content of its own", () => {
      // The carrier is a message whose ONLY part is the compaction request. A
      // message with text in it is the user talking, and dropping it would
      // delete what they said.
      expect(
        SessionPrompt.isOverflowCompactionCarrier(
          {
            info: { role: "user", model: MODEL_A },
            parts: [{ type: "text" }, { type: "compaction", overflow: true }],
          },
          MODEL_B,
        ),
      ).toBe(false)
      expect(
        SessionPrompt.isOverflowCompactionCarrier({ info: { role: "user", model: MODEL_A }, parts: [] }, MODEL_B),
      ).toBe(false)
      expect(
        SessionPrompt.isOverflowCompactionCarrier(
          { info: { role: "assistant" }, parts: [{ type: "compaction", overflow: true }] },
          MODEL_B,
        ),
      ).toBe(false)
    })
  })

  describe("SessionPrompt.findStaleOverflowCompactions", () => {
    const MODEL_A = { providerID: "openai", modelID: "gpt-5.4-mini" }
    const MODEL_B = { providerID: "openai", modelID: "gpt-5.6-luna" }
    const carrier = (id: string) => ({
      info: { id, role: "user", model: MODEL_A },
      parts: [{ type: "compaction", overflow: true }],
    })
    const summaryFor = (id: string, over: Partial<{ summary: boolean; finish: string; error: unknown }> = {}) => ({
      info: { id: `${id}-summary`, role: "assistant", parentID: id, summary: true, finish: "stop", ...over },
      parts: [{ type: "text" }],
    })
    const user = (id: string) => ({
      info: { id, role: "user", model: MODEL_A },
      parts: [{ type: "text" }],
    })

    test("finds a queued carrier once the model changed", () => {
      const stale = SessionPrompt.findStaleOverflowCompactions([user("u1"), carrier("c1")], MODEL_B)
      expect(stale).toEqual([{ id: "c1", queuedModel: MODEL_A }])
    })

    test("leaves a carrier whose compaction ALREADY RAN, model change or not", () => {
      // `filterCompacted` cuts the history AT that carrier and returns it as
      // msgs[0], with the summary right behind it. Remove it and the window
      // opens on an assistant message — Anthropic rejects the request outright
      // ("first message must use the 'user' role") for every later turn.
      const msgs = [carrier("c1"), summaryFor("c1"), user("u2")]
      expect(SessionPrompt.findStaleOverflowCompactions(msgs, MODEL_B)).toEqual([])
    })

    test("an unfinished or errored summary does not count as processed", () => {
      // Same predicate `filterCompacted` uses — the two must never disagree
      // about which carrier is spent.
      for (const over of [
        { finish: undefined as unknown as string },
        { error: { name: "APIError" } },
        { summary: false },
      ]) {
        const stale = SessionPrompt.findStaleOverflowCompactions(
          [carrier("c1"), summaryFor("c1", over), user("u2")],
          MODEL_B,
        )
        expect(stale.map((entry) => entry.id)).toEqual(["c1"])
      }
    })

    test("a summary belonging to another carrier does not protect this one", () => {
      const msgs = [carrier("c0"), summaryFor("c0"), user("u1"), carrier("c1")]
      expect(SessionPrompt.findStaleOverflowCompactions(msgs, MODEL_B).map((entry) => entry.id)).toEqual(["c1"])
    })

    test("nothing is stale while the model is unchanged", () => {
      expect(SessionPrompt.findStaleOverflowCompactions([user("u1"), carrier("c1")], MODEL_A)).toEqual([])
    })
  })

  describe("SessionPrompt.scanTurn", () => {
    const user = (id: string, parts: { type: string; overflow?: boolean }[] = [{ type: "text" }]) =>
      ({ info: { id, role: "user", model: { providerID: "openai", modelID: "gpt-5.6-luna" } }, parts }) as never
    const assistant = (id: string, finish?: string) =>
      ({ info: { id, role: "assistant", ...(finish ? { finish } : {}) }, parts: [] }) as never

    test("reports the last user/assistant positions, not their ID order", () => {
      const scan = SessionPrompt.scanTurn([user("u1"), assistant("a1", "stop"), user("u2")])
      expect(String(scan.lastUser?.id)).toBe("u2")
      expect(scan.lastUserIdx).toBe(2)
      expect(scan.lastAssistantIdx).toBe(1)
      expect(scan.lastFinishedIdx).toBe(1)
    })

    test("collects only the tasks newer than the last finished assistant message", () => {
      const scan = SessionPrompt.scanTurn([
        user("u1", [{ type: "compaction", overflow: true }]),
        assistant("a1", "stop"),
        user("u2", [{ type: "compaction", overflow: true }]),
      ])
      expect(scan.tasks).toHaveLength(1)
    })

    test("re-scanning a list with the carrier removed moves lastUser back to the real message", () => {
      // This is what the loop does after dropping a stale carrier: the request
      // that overflowed becomes the message being answered again.
      const msgs = [user("u1"), assistant("a1"), user("carrier", [{ type: "compaction", overflow: true }])]
      expect(String(SessionPrompt.scanTurn(msgs).lastUser?.id)).toBe("carrier")
      const withoutCarrier = msgs.filter((m) => (m as unknown as { info: { id: string } }).info.id !== "carrier")
      const rescan = SessionPrompt.scanTurn(withoutCarrier)
      expect(String(rescan.lastUser?.id)).toBe("u1")
      expect(rescan.tasks).toHaveLength(0)
    })
  })

  describe("SessionCompaction.continuePrompt", () => {
    test("says nothing about overflow on a routine compaction", () => {
      const text = SessionCompaction.continuePrompt({ overflow: false, hadMedia: false })
      expect(text).not.toContain("exceeded")
      expect(text).toContain("Continue if you have next steps")
    })

    test("does not invent attachments when the history had none", () => {
      // The production bug: sessions with zero attachments were told the request
      // failed "due to large media attachments", so the agent apologised to the
      // user for images that never existed.
      const text = SessionCompaction.continuePrompt({ overflow: true, hadMedia: false })
      expect(text).not.toContain("media")
      expect(text).not.toContain("attachments were too large")
      expect(text).toContain("exceeded this model's context window")
      expect(text).toContain("No attachments were involved")
    })

    test("points at the causes a compaction cannot fix", () => {
      const text = SessionCompaction.continuePrompt({ overflow: true, hadMedia: false })
      expect(text).toContain("tools/MCP servers")
      expect(text).toContain("context window is too small")
    })

    test("keeps the media guidance when media really was in context", () => {
      const text = SessionCompaction.continuePrompt({ overflow: true, hadMedia: true })
      expect(text).toContain("media files were removed from context")
      expect(text).toContain("smaller or fewer files")
    })

    test("always ends with the continue instruction", () => {
      for (const overflow of [true, false]) {
        for (const hadMedia of [true, false]) {
          expect(SessionCompaction.continuePrompt({ overflow, hadMedia })).toEndWith(
            "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.",
          )
        }
      }
    })
  })
})
