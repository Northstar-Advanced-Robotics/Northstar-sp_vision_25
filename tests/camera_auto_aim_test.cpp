#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/stats.hpp"

// Same detect+track pipeline as auto_aim_test, but reading live frames from
// io::Camera (any camera_name it supports -- mindvision/hikrobot/usb)
// instead of a recorded .avi/.txt pair. There's no live gimbal/IMU here, so
// solver's R_gimbal2world stays at its default identity (world frame ==
// camera/gimbal frame) -- fine for eyeballing detect/track quality on a
// bench-mounted camera, not a substitute for auto_aim_test's real
// ground-truth-driven accuracy numbers.

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明 }"
  "{config-path c  | configs/camera.yaml | yaml配置文件的路径}"
  "{headless       |                     | "
  "完全不使用GUI（无imshow/waitKey，不需要任何X服务器），用于纯SSH压测}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");
  bool headless_mode = cli.has("headless");

  tools::Exiter exiter;

  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, !headless_mode);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);

  tools::Stats detect_stats{"detect"};
  tools::Stats track_stats{"track"};

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();

  for (int frame_count = 0; !exiter.exit(); frame_count++) {
    camera.read(img, timestamp);
    if (img.empty()) {
      frame_count--;
      continue;
    }

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;

    auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto detect_done = std::chrono::steady_clock::now();
    detect_stats.add(tools::delta_time(detect_done, detect_start) * 1e3, frame_count);

    auto targets = tracker.track(armors, timestamp);
    auto track_done = std::chrono::steady_clock::now();
    track_stats.add(tools::delta_time(track_done, detect_done) * 1e3, frame_count);

    tools::logger()->info(
      "[{}] {:.1f} fps, armors: {}, targets: {}, detect: {:.1f}ms, track: {:.1f}ms", frame_count,
      1 / dt, armors.size(), targets.size(),
      tools::delta_time(detect_done, detect_start) * 1e3,
      tools::delta_time(track_done, detect_done) * 1e3);

    if (!targets.empty()) {
      const auto & target = targets.front();
      Eigen::VectorXd x = target.ekf_x();
      tools::logger()->info(
        "  target {}: x={:.2f} y={:.2f} z={:.2f} yaw={:.1f}deg", target.last_id, x[0], x[2], x[4],
        x[6] * 57.3);
    }

    if (headless_mode) continue;

    for (const auto & target : targets) {
      for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }
    }

    tools::draw_text(
      img, fmt::format("armors:{} targets:{}", armors.size(), targets.size()), {10, 30},
      {154, 50, 205});

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("camera_auto_aim_test", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
