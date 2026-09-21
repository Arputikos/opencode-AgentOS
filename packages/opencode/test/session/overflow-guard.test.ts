import { describe, expect, test } from "bun:test"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionPrompt } from "../../src/session/prompt"

// Pure decision logic behind the two overflow patches. Kept separate from
// `compaction.test.ts` because those tests need a real temp directory and this
// file deliberately needs nothing at all.
describe("session overflow guard", () => {
  describe("SessionPrompt.nextOverflowCompaction", () => {
    test("does not count our own pre-emptive compactions", () => {
      // Token accounting decided a compaction is due — always safe, always
      // shrinks the next request. Must never trip the breaker, no matter how
      // often a long session does it.
      let state = { consecutive: 0, giveUp: false }
      for (let i = 0; i < 50; i++) {
        state = SessionPrompt.nextOverflowCompaction({ rejected: false, consecutive: state.consecutive })
        expect(state.giveUp).toBe(false)
        expect(state.consecutive).toBe(0)
      }
    })

    test("allows exactly one compaction in response to a provider rejection", () => {
      const first = SessionPrompt.nextOverflowCompaction({ rejected: true, consecutive: 0 })
      expect(first).toEqual({ consecutive: 1, giveUp: false })
    })

    test("gives up when a rejection follows the compaction that was meant to fix it", () => {
      const second = SessionPrompt.nextOverflowCompaction({ rejected: true, consecutive: 1 })
      expect(second.giveUp).toBe(true)
      expect(second.consecutive).toBe(2)
    })

    test("an accepted request in between resets the breaker", () => {
      // Rejection → compaction → the request goes through → later, a fresh
      // rejection. That is not a loop, so the second rejection gets its own
      // compaction rather than being treated as the doomed retry.
      const rejected = SessionPrompt.nextOverflowCompaction({ rejected: true, consecutive: 0 })
      const accepted = SessionPrompt.nextOverflowCompaction({ rejected: false, consecutive: rejected.consecutive })
      expect(accepted.consecutive).toBe(0)
      const again = SessionPrompt.nextOverflowCompaction({ rejected: true, consecutive: accepted.consecutive })
      expect(again.giveUp).toBe(false)
    })

    test("terminates — the production loop ran 8 compactions in 45s", () => {
      // Drive it the way runLoop does and assert it stops. Without the guard
      // this is an infinite sequence.
      let consecutive = 0
      let compactions = 0
      for (let i = 0; i < 100; i++) {
        const next = SessionPrompt.nextOverflowCompaction({ rejected: true, consecutive })
        consecutive = next.consecutive
        if (next.giveUp) break
        compactions++
      }
      expect(compactions).toBe(1)
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
