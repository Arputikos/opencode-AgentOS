import type { Config } from "@/config/config"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import type { MessageV2 } from "./message-v2"

const COMPACTION_BUFFER = 20_000

/**
 * The input size at which a session compacts pre-emptively; undefined when the
 * model's window is unknown (`limit.context = 0`, e.g. a model from the config
 * without a `limit`) — there is no line to compact at.
 *
 * Patched (agentos): split out of `isOverflow` so the request footprint can
 * ask the same question about its fixed overhead — if the system prompt and
 * the tool schemas alone reach this line, no amount of compaction can get a
 * request under it (see `SessionPrompt` → `requestCompaction`).
 */
export function compactionThreshold(input: { cfg: Config.Info; model: Provider.Model }): number | undefined {
  if (input.model.limit.context === 0) return undefined
  const reserved =
    input.cfg.compaction?.reserved ?? Math.min(COMPACTION_BUFFER, ProviderTransform.maxOutputTokens(input.model))
  return input.model.limit.input
    ? input.model.limit.input - reserved
    : input.model.limit.context - ProviderTransform.maxOutputTokens(input.model)
}

export function isOverflow(input: { cfg: Config.Info; tokens: MessageV2.Assistant["tokens"]; model: Provider.Model }) {
  if (input.cfg.compaction?.auto === false) return false
  const threshold = compactionThreshold(input)
  if (threshold === undefined) return false

  const count =
    input.tokens.total || input.tokens.input + input.tokens.output + input.tokens.cache.read + input.tokens.cache.write

  return count >= threshold
}
