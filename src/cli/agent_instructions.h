#pragma once

namespace slopfab::cli {
// In agent mode, create or atomically upgrade working-directory instructions.
// The first line is a numeric version; unversioned files are version zero.
// Equal/newer versions and symlinks are preserved. Emit context after a write.
void ensure_agent_instructions();
}
