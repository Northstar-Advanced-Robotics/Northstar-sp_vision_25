#ifndef TOOLS__STATS_HPP
#define TOOLS__STATS_HPP

#include <string>
#include <vector>

namespace tools
{
// Running mean/stddev/min/max (Welford's online algorithm, so this doesn't
// need to keep every sample around) plus the worst_n highest-value samples
// seen, each tagged with the frame index that produced it -- for spotting
// *which* frames are the outliers, not just that variance exists. Call
// add() once per frame; a summary (count/mean/stddev/min/max + the sorted
// worst_n list) auto-logs via tools::logger() every report_every samples,
// and once more from the destructor so short runs still get a final report
// even if they don't land on a report_every boundary.
class Stats
{
public:
  explicit Stats(std::string name, int report_every = 200, int worst_n = 5);
  ~Stats();

  Stats(const Stats &) = delete;
  Stats & operator=(const Stats &) = delete;

  void add(double value_ms, int frame_index = -1);
  void log_summary() const;

  // For deriving cross-metric values (e.g. a throughput ceiling from two
  // different Stats objects' means) that don't fit this class's own
  // single-metric summary line.
  double mean() const { return mean_; }
  long count() const { return count_; }

private:
  std::string name_;
  int report_every_;
  int worst_n_;
  long count_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;  // Welford's running sum of squared differences from the mean
  double min_ms_ = 0.0;
  double max_ms_ = 0.0;
  bool logged_since_last_add_ = false;

  struct Sample
  {
    double value_ms;
    int frame_index;
  };
  std::vector<Sample> worst_;  // kept sorted ascending by value_ms, size <= worst_n_
};

}  // namespace tools

#endif  // TOOLS__STATS_HPP
