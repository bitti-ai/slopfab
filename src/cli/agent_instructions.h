#pragma once

namespace slopfab::cli {
// Bootstrap the working directory only in agent mode. Existing files, including
// user-authored instructions, are never replaced or repeated in the response.
void ensure_agent_instructions();
}
