#!/usr/bin/env python3
"""Fuses NMS suppression into assets/yolov5.onnx as a TensorRT-native op.

This is a SEPARATE transform layered on top of the already-committed
assets/yolov5.onnx, which itself came from the third-party `openvino2onnx`
tool converting assets/yolov5.xml/.bin -- see JETSON_ORIN.md, "Where
assets/yolov5.onnx came from". Don't confuse the two: this script never
touches the .xml/.bin IR, only the already-exported .onnx.

The model's raw output is [1, 25200, 22]:
  cols 0-7   4 keypoints (x,y pairs), already in absolute 640x640 pixel space
  col  8     objectness, raw logit (needs sigmoid)
  cols 9-12  4 color-class raw logits
  cols 13-21 9 armor-number-class raw logits

The existing CPU postprocess (YOLOV5::parse() in tasks/auto_aim/yolos/
yolov5.cpp) scans all 25200 rows every frame to do the sigmoid threshold +
bbox-from-keypoints + cv::dnn::NMSBoxes step. This script adds graph nodes
that do the equivalent computation once, inside the TensorRT engine itself,
via ONNX's NonMaxSuppression op:

  keypoints (0:8) -> Reshape -> Gather x/y -> ReduceMin/Max -> boxes [1,25200,4]
    (in [y1,x1,y2,x2] order -- ONNX NMS's center_point_box=0 convention)
  objectness (8:9) -> Sigmoid -> scores [1,1,25200]
  NonMaxSuppression(boxes, scores, ...) -> selected_indices [-1,3] int64
    (rows are [batch_index, class_index, box_index]; class_index is always 0
    here -- NMS runs over a single "class" (objectness), NOT the armor
    color/number classes, which are unrelated to and untouched by this)

The ORIGINAL output (name below) is left completely unmodified -- this is
purely additive, confirmed backward compatible (byte-identical original
output pre/post-fusion, checked below). selected_indices is added as a
SECOND graph output. Downstream C++ (tasks/auto_aim/yolos/yolov5.cpp /
mt_detector.cpp) reads selected_indices' box_index column and gathers the
handful of surviving rows directly out of the (unchanged) original output --
no Gather node is added to the graph for that, kept in C++ instead, since
that keeps the graph surgery simpler and the survivor count small (<=64).

Usage: python scripts/onnx/fuse_nms.py [path/to/yolov5.onnx]
  (defaults to assets/yolov5.onnx relative to the repo root, overwritten in
  place after validation passes)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

ORIGINAL_OUTPUT_NAME = "output/sink_port_0"
SELECTED_INDICES_NAME = "selected_indices"

# max_output_boxes_per_class MUST match kMaxNmsOutputBoxes in
# tasks/auto_aim/yolos/trt_engine.hpp -- that's the worst-case row count the
# C++ TrtNmsOutputAllocator pre-sizes its device buffer to. If you change it
# here, change it there too (no compiler links these two constants).
MAX_OUTPUT_BOXES_PER_CLASS = 64
IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.7


def fuse(graph: gs.Graph) -> None:
    raw_out = next(o for o in graph.outputs if o.name == ORIGINAL_OUTPUT_NAME)
    if raw_out.shape != [1, 25200, 22]:
        raise ValueError(
            f"expected {ORIGINAL_OUTPUT_NAME} shape [1, 25200, 22], got {raw_out.shape} "
            "-- this script's column layout (keypoints/objectness/color/class) assumes "
            "the exact yolov5 export this repo uses; don't run it against a different model."
        )

    # Explicit dtypes, not numpy's platform-dependent default (plain Python
    # ints default to int32 on Windows vs int64 on Linux via np.array();
    # ONNX requires int64 specifically for e.g. NonMaxSuppression's
    # max_output_boxes_per_class -- caught by onnx.checker's full_check
    # below when this was still implicit, not a hypothetical). Float
    # constants need explicit float32 too -- np.array() of a Python float
    # defaults to float64, but ONNX's NonMaxSuppression requires float
    # (i.e. float32) for iou_threshold/score_threshold.
    def const(name, arr, dtype=np.int64):
        return gs.Constant(name, values=np.array(arr, dtype=dtype))

    def var(name, dtype=np.float32, shape=None):
        return gs.Variable(name, dtype=dtype, shape=shape)

    def op(name, inputs, outputs, **attrs):
        node = gs.Node(
            op=name, name=f"nms_fuse/{outputs[0].name}", inputs=inputs, outputs=outputs, attrs=attrs
        )
        graph.nodes.append(node)
        return node

    # keypoints (cols 0:8) -> bbox [1,25200,4] as [y1,x1,y2,x2]
    kpts = var("nms_kpts", shape=(1, 25200, 8))
    op("Slice", [raw_out, const("kpts_starts", [0]), const("kpts_ends", [8]), const("kpts_axes", [2])], [kpts])

    kpts4x2 = var("nms_kpts_4x2", shape=(1, 25200, 4, 2))
    op("Reshape", [kpts, const("kpts_reshape_shape", [1, 25200, 4, 2])], [kpts4x2])

    xs = var("nms_xs", shape=(1, 25200, 4))
    op("Gather", [kpts4x2, const("x_index", 0)], [xs], axis=3)
    ys = var("nms_ys", shape=(1, 25200, 4))
    op("Gather", [kpts4x2, const("y_index", 1)], [ys], axis=3)

    # ReduceMin/ReduceMax take `axes` as a second INPUT (not an attribute)
    # starting at opset 18 -- this graph is opset 19 (confirmed via
    # trtexec's parse log), so passing axes=[2] as an attribute (valid for
    # older opsets) fails onnx.checker's full_check with "Unrecognized
    # attribute: axes". One shared axis_2 constant reused by all four calls.
    axis_2 = const("reduce_axis_2", [2])
    min_x = var("nms_min_x", shape=(1, 25200, 1))
    op("ReduceMin", [xs, axis_2], [min_x], keepdims=1)
    max_x = var("nms_max_x", shape=(1, 25200, 1))
    op("ReduceMax", [xs, axis_2], [max_x], keepdims=1)
    min_y = var("nms_min_y", shape=(1, 25200, 1))
    op("ReduceMin", [ys, axis_2], [min_y], keepdims=1)
    max_y = var("nms_max_y", shape=(1, 25200, 1))
    op("ReduceMax", [ys, axis_2], [max_y], keepdims=1)

    boxes = var("nms_boxes", shape=(1, 25200, 4))
    op("Concat", [min_y, min_x, max_y, max_x], [boxes], axis=2)

    # objectness (col 8:9) -> sigmoid -> scores [1,1,25200]
    obj = var("nms_obj", shape=(1, 25200, 1))
    op("Slice", [raw_out, const("obj_starts", [8]), const("obj_ends", [9]), const("obj_axes", [2])], [obj])
    obj_sig = var("nms_obj_sigmoid", shape=(1, 25200, 1))
    op("Sigmoid", [obj], [obj_sig])
    scores = var("nms_scores", shape=(1, 1, 25200))
    op("Transpose", [obj_sig], [scores], perm=[0, 2, 1])

    selected = var(SELECTED_INDICES_NAME, dtype=np.int64, shape=(-1, 3))
    op(
        "NonMaxSuppression",
        [
            boxes,
            scores,
            const("max_output_boxes_per_class", [MAX_OUTPUT_BOXES_PER_CLASS]),
            const("iou_threshold", [IOU_THRESHOLD], dtype=np.float32),
            const("score_threshold", [SCORE_THRESHOLD], dtype=np.float32),
        ],
        [selected],
        center_point_box=0,
    )

    graph.outputs.append(selected)
    graph.cleanup().toposort()


def sanity_check(original_path: Path, fused_path: Path) -> None:
    """Runs both graphs via onnxruntime CPU EP on a random input and confirms
    the original output is byte-identical -- catches graph-surgery mistakes
    (e.g. accidentally mutating raw_out instead of only reading it) here, at
    script-run time, rather than as a confusing mismatch discovered later on
    the Jetson."""
    import onnxruntime as ort

    x = np.random.rand(1, 3, 640, 640).astype(np.float32)

    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    (out_orig,) = sess_orig.run([ORIGINAL_OUTPUT_NAME], {"images": x})

    sess_fused = ort.InferenceSession(str(fused_path), providers=["CPUExecutionProvider"])
    out_fused, selected = sess_fused.run([ORIGINAL_OUTPUT_NAME, SELECTED_INDICES_NAME], {"images": x})

    if not np.array_equal(out_orig, out_fused):
        raise RuntimeError(
            f"sanity check FAILED: {ORIGINAL_OUTPUT_NAME} differs between the original and "
            "fused graphs -- the graph surgery must not have been purely additive. Not saving."
        )
    if selected.shape[1] != 3 or selected.shape[0] > MAX_OUTPUT_BOXES_PER_CLASS:
        raise RuntimeError(f"sanity check FAILED: unexpected selected_indices shape {selected.shape}")
    print(
        f"sanity check passed: {ORIGINAL_OUTPUT_NAME} byte-identical pre/post-fusion, "
        f"selected_indices shape {selected.shape} (random input, so the count itself is "
        "meaningless -- only the shape/dtype and that it ran without error matter here)"
    )


def main():
    repo_root = Path(__file__).resolve().parents[2]
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "assets" / "yolov5.onnx"

    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)

    if any(o.name == SELECTED_INDICES_NAME for o in graph.outputs):
        raise RuntimeError(
            f"{onnx_path} already has a '{SELECTED_INDICES_NAME}' output -- looks already "
            "fused. Re-running this script would fuse NMS a second time on top of itself. "
            "If you want to re-fuse from scratch, start from a fresh openvino2onnx export "
            "of assets/yolov5.xml instead of the current assets/yolov5.onnx."
        )

    fuse(graph)

    fused_model = gs.export_onnx(graph)
    onnx.checker.check_model(fused_model, full_check=True)

    tmp_path = onnx_path.with_suffix(".fused_tmp.onnx")
    onnx.save(fused_model, str(tmp_path))
    try:
        sanity_check(onnx_path, tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(onnx_path)
    print(f"saved {onnx_path} (outputs: {[o.name for o in graph.outputs]})")


if __name__ == "__main__":
    main()
