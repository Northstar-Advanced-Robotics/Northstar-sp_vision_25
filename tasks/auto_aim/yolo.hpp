#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "armor.hpp"

namespace auto_aim
{
class YOLOBase
{
public:
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;

  // Alternate to postprocess() for backends that run NMS fused into the
  // inference engine itself (currently: YOLOV5 + device: TENSORRT only --
  // see trt_engine.hpp/scripts/onnx/fuse_efficient_nms.py, TensorRT's
  // EfficientNMS_TRT plugin). detection_scores/num_detections come from the
  // engine's fixed-shape (padded to kMaxNmsOutputBoxes) outputs -- NMS
  // suppression has already happened on the GPU, so no cv::dnn::NMSBoxes
  // call is needed here; the implementation recovers which raw_output row
  // each detection came from by matching detection_scores against
  // raw_output's own objectness column.
  //
  // Default throws rather than = 0: YOLOV8/YOLO11 (OpenVINO-only, no
  // TensorRT support at all) never override this and it's never reached in
  // practice (setting device: TENSORRT with yolo_name: yolov8/yolo11
  // already fails earlier, at OpenVINO device-string validation) -- kept as
  // fail-loud defense-in-depth rather than requiring those classes to carry
  // a meaningless empty override.
  virtual std::list<Armor> postprocess_from_efficient_nms(
    double /*scale*/, cv::Mat & /*raw_output*/, const float * /*detection_scores*/,
    int /*num_detections*/, const cv::Mat & /*bgr_img*/, int /*frame_count*/)
  {
    throw std::runtime_error(
      "YOLOBase: postprocess_from_efficient_nms not supported by this backend "
      "(only YOLOV5 + device: TENSORRT implements the fused-NMS path)");
  }
};

class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  std::list<Armor> postprocess_from_efficient_nms(
    double scale, cv::Mat & raw_output, const float * detection_scores, int num_detections,
    const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP