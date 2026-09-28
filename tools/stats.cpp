#include "stats.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"

namespace tools
{
Stats::Stats(std::string name, int report_every, int worst_n)
: name_(std::move(name)), report_every_(report_every), worst_n_(worst_n)
{
  worst_.reserve(static_cast<size_t>(worst_n_));
}

Stats::~Stats()
{
  // Only log a final summary if add() was actually called since the last
  // periodic report -- avoids a redundant duplicate line when the last
  // add() happened to land exactly on a report_every_ boundary, and avoids
  // logging an empty/all-zero summary for a Stats object that never saw
  // any samples at all (e.g. a backend that was constructed but never used).
  if (count_ > 0 && !logged_since_last_add_) log_summary();
}

void Stats::add(double value_ms, int frame_index)
{
  count_++;
  double delta = value_ms - mean_;
  mean_ += delta / static_cast<double>(count_);
  double delta2 = value_ms - mean_;
  m2_ += delta * delta2;

  if (count_ == 1) {
    min_ms_ = max_ms_ = value_ms;
  } else {
    min_ms_ = std::min(min_ms_, value_ms);
    max_ms_ = std::max(max_ms_, value_ms);
  }

  if (static_cast<int>(worst_.size()) < worst_n_ || value_ms > worst_.front().value_ms) {
    auto it = std::lower_bound(
      worst_.begin(), worst_.end(), value_ms,
      [](const Sample & s, double v) { return s.value_ms < v; });
    worst_.insert(it, Sample{value_ms, frame_index});
    if (static_cast<int>(worst_.size()) > worst_n_) worst_.erase(worst_.begin());
  }

  logged_since_last_add_ = false;
  if (report_every_ > 0 && count_ % report_every_ == 0) log_summary();
}

void Stats::log_summary() const
{
  double variance = count_ > 1 ? m2_ / static_cast<double>(count_ - 1) : 0.0;
  double stddev = std::sqrt(variance);

  std::string worst_str;
  for (auto it = worst_.rbegin(); it != worst_.rend(); ++it) {
    if (!worst_str.empty()) worst_str += ", ";
    if (it->frame_index >= 0) {
      worst_str += fmt::format("{:.2f}ms@{}", it->value_ms, it->frame_index);
    } else {
      worst_str += fmt::format("{:.2f}ms", it->value_ms);
    }
  }

  tools::logger()->info(
    "[STATS] {}: n={} mean={:.2f}ms stddev={:.2f}ms min={:.2f}ms max={:.2f}ms worst=[{}]", name_,
    count_, mean_, stddev, min_ms_, max_ms_, worst_str);

  // const_cast is safe here (doesn't touch any const member): the flag is
  // purely an implementation detail letting the destructor skip a
  // redundant final log after add() itself already logged one on this
  // exact sample, not part of this object's logical/observable state.
  const_cast<Stats *>(this)->logged_since_last_add_ = true;
}

}  // namespace tools
