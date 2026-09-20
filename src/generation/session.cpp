#include "session_state.h"
#include <mutex>
#include <stdexcept>

namespace slopfab {
struct GenerationSession::Impl {
  std::mutex mutex;
  generation::ReusedGenerationModels models;
};
namespace {
std::mutex execution_mutex;
GenerationSession& default_session() {
  static GenerationSession session;
  return session;
}
}
GenerationSession::GenerationSession() : impl_(std::make_unique<Impl>()) {}
GenerationSession::~GenerationSession() = default;
void GenerationSession::clear() {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) throw std::runtime_error("generation session is busy");
  impl_->models.clear();
}
RunResult run_generate(GenerationSession& session, const GenerateRequest& request,
                       const GeneratePlan& plan, const RunOptions& options) {
  std::unique_lock<std::mutex> session_lock(session.impl_->mutex, std::try_to_lock);
  std::unique_lock<std::mutex> execution_lock(execution_mutex, std::try_to_lock);
  if (!session_lock.owns_lock() || !execution_lock.owns_lock()) {
    RunResult result;
    result.message = "generation execution is busy";
    return result;
  }
  return generation::run_generate_impl(request, plan, options, session.impl_->models);
}
RunResult run_generate(const GenerateRequest& request, const GeneratePlan& plan,
                       const RunOptions& options) {
  return run_generate(default_session(), request, plan, options);
}
void clear_reused_generation_models() { default_session().clear(); }
}  // namespace slopfab
