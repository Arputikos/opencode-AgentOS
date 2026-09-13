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
