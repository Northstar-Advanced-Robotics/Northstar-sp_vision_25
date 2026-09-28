#!/usr/bin/env python3
"""Fuses NMS into assets/yolov5.onnx via TensorRT's EfficientNMS_TRT plugin.

This SUPERSEDES the earlier scripts/onnx/fuse_nms.py approach (ONNX-standard
NonMaxSuppression -> a dynamically-shaped `selected_indices` output). That
approach forced TensorRT's data-dependent-shape (DDS) machinery
(IOutputAllocator), which turned out to make enqueueV3() a blocking call
(TensorRT 10.0-10.7 has a documented, NVIDIA-acknowledged performance
regression for DDS networks -- fixed in 10.8.0, release notes: "Fixed
performance regression for TensorRT 10.x compared to TensorRT 8.6 for
networks involving data-dependent shapes (for example, non-max suppression
or non-zero operations)". This Jetson is pinned to TensorRT 10.3.0 via
JetPack 6.1/6.2 (both ship 10.3, no in-place upgrade available), so instead
of waiting on that fix, this script sidesteps DDS entirely.

TensorRT's own EfficientNMS_TRT plugin (confirmed present in this Jetson's
plugin registry: `trt.get_plugin_registry().plugin_creator_list`) outputs
FIXED, statically-shaped tensors padded to max_output_boxes --
num_detections [1,1] int32, detection_boxes/scores/classes
[1,max_output_boxes,...] -- so no IOutputAllocator is needed at all, and
enqueueV3() goes back to being a cheap dispatch call.

Trade-off: EfficientNMS_TRT's outputs don't include the original row index
into the raw [1,25200,22] output, only derived box/score/class. But this
model's downstream code needs the original row's 4 keypoints (not just an
axis-aligned box) for the armor-corner PnP solve. So downstream C++
recovers the original row by matching detection_scores against the
objectness column it already reads back every frame (deterministic sigmoid,
exact float match) -- see YOLOV5::parse_from_efficient_nms_outputs in
tasks/auto_aim/yolos/yolov5.cpp.

Run against a CLEAN (never-fused) yolov5.onnx -- this script's idempotency
guard refuses a graph that already has an EfficientNMS_TRT node, but does
NOT know how to strip a previous fuse_nms.py (DDS) fusion back out. Use
scripts/onnx/yolov5_prefusion_backup.onnx (checked in alongside this
script, extracted from git history at commit 8b609bc~1, the last commit
before any NMS fusion) as the starting point if assets/yolov5.onnx is
already DDS-fused.

Usage: python scripts/onnx/fuse_efficient_nms.py [path/to/yolov5.onnx]
  (defaults to assets/yolov5.onnx relative to the repo root, overwritten in
  place after validation passes)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

ORIGINAL_OUTPUT_NAME = "output/sink_port_0"

# Must match kMaxNmsOutputBoxes in tasks/auto_aim/yolos/trt_engine.hpp --
# that's the row count the C++ side pre-sizes its fixed output buffers to.
# No compiler links these two constants, so keep them in sync by hand.
MAX_OUTPUT_BOXES = 64
IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.7

EFFICIENT_NMS_OUTPUT_NAMES = [
    "num_detections",
    "detection_boxes",
    "detection_scores",
    "detection_classes",
]


def fuse(graph: gs.Graph) -> None:
    raw_out = next(o for o in graph.outputs if o.name == ORIGINAL_OUTPUT_NAME)
    if raw_out.shape != [1, 25200, 22]:
        raise ValueError(
            f"expected {ORIGINAL_OUTPUT_NAME} shape [1, 25200, 22], got {raw_out.shape} "
            "-- this script's column layout (keypoints/objectness/color/class) assumes "
            "the exact yolov5 export this repo uses; don't run it against a different model."
        )

    # Explicit dtypes, not numpy's platform-dependent default -- see the
    # long comment in fuse_nms.py's `const()` for why this matters (a real
    # Windows-vs-Linux bug hit during that script's development).
    def const(name, arr, dtype=np.int64):
        return gs.Constant(name, values=np.array(arr, dtype=dtype))

    def var(name, dtype=np.float32, shape=None):
        return gs.Variable(name, dtype=dtype, shape=shape)

    def op(name, inputs, outputs, **attrs):
        node = gs.Node(
            op=name, name=f"effnms_fuse/{outputs[0].name}", inputs=inputs, outputs=outputs, attrs=attrs
        )
        graph.nodes.append(node)
        return node

    # keypoints (cols 0:8) -> boxes [1,25200,4]. EfficientNMS_TRT's
    # box_coding=0 (BoxCorner) only requires a consistent (min,min,max,max)
    # layout -- x/y order doesn't matter for IOU math, area overlap is
    # symmetric under axis relabeling. Same construction as fuse_nms.py.
    kpts = var("nms_kpts", shape=(1, 25200, 8))
    op("Slice", [raw_out, const("kpts_starts", [0]), const("kpts_ends", [8]), const("kpts_axes", [2])], [kpts])

    kpts4x2 = var("nms_kpts_4x2", shape=(1, 25200, 4, 2))
    op("Reshape", [kpts, const("kpts_reshape_shape", [1, 25200, 4, 2])], [kpts4x2])

    xs = var("nms_xs", shape=(1, 25200, 4))
    op("Gather", [kpts4x2, const("x_index", 0)], [xs], axis=3)
    ys = var("nms_ys", shape=(1, 25200, 4))
    op("Gather", [kpts4x2, const("y_index", 1)], [ys], axis=3)

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

    # objectness (col 8:9) -> scores [1,25200,1] (EfficientNMS wants
    # [batch,num_boxes,num_classes] -- NOT the [batch,num_classes,num_boxes]
    # ONNX's own NonMaxSuppression op used, and no Transpose needed here).
    # Raw logit fed straight in with score_activation=True below, so the
    # plugin applies sigmoid internally -- one fewer node than fuse_nms.py,
    # which had to do its own Sigmoid because the ONNX NMS op has no
    # equivalent activation option.
    scores = var("nms_scores", shape=(1, 25200, 1))
    op(
        "Slice",
        [raw_out, const("obj_starts", [8]), const("obj_ends", [9]), const("obj_axes", [2])],
        [scores],
    )

    num_detections = var("num_detections", dtype=np.int32, shape=(1, 1))
    detection_boxes = var("detection_boxes", shape=(1, MAX_OUTPUT_BOXES, 4))
    detection_scores = var("detection_scores", shape=(1, MAX_OUTPUT_BOXES))
    detection_classes = var("detection_classes", dtype=np.int32, shape=(1, MAX_OUTPUT_BOXES))

    node = gs.Node(
        op="EfficientNMS_TRT",
        name="effnms_fuse/EfficientNMS_TRT",
        inputs=[boxes, scores],
        outputs=[num_detections, detection_boxes, detection_scores, detection_classes],
        attrs={
            "plugin_version": "1",
            "background_class": -1,
            "max_output_boxes": MAX_OUTPUT_BOXES,
            "score_threshold": SCORE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "score_activation": True,
            "class_agnostic": True,
            "box_coding": 0,
        },
    )
    graph.nodes.append(node)

    graph.outputs.extend([num_detections, detection_boxes, detection_scores, detection_classes])
    graph.cleanup().toposort()


def sanity_check(original_path: Path, graph: gs.Graph) -> None:
    """EfficientNMS_TRT is a TensorRT-proprietary op -- no onnxruntime CPU EP
    kernel exists for it, so (unlike fuse_nms.py's ONNX-standard
    NonMaxSuppression) we can't just load the fused graph in onnxruntime and
    run it end to end; that would fail on the custom node before it ever
    got to compare outputs. Instead: build a throwaway DEBUG graph that's
    byte-for-byte the same up through `boxes`/`scores` (the only genuinely
    NEW graph surgery here -- box_coding/attrs on the NMS node itself can
    only really be validated by building it in TensorRT, done separately on
    the Jetson) but stops there instead of feeding into EfficientNMS_TRT.
    Confirms the original output is untouched and boxes/scores come out
    sane-shaped and sane-valued."""
    import onnxruntime as ort

    debug_graph = graph.copy()
    boxes = next(t for t in debug_graph.tensors().values() if t.name == "nms_boxes")
    scores = next(t for t in debug_graph.tensors().values() if t.name == "nms_scores")
    debug_graph.outputs = [
        next(o for o in debug_graph.outputs if o.name == ORIGINAL_OUTPUT_NAME),
        boxes,
        scores,
    ]
    debug_graph.cleanup().toposort()
    debug_model = gs.export_onnx(debug_graph)
    debug_path = original_path.with_suffix(".debug_tmp.onnx")
    onnx.save(debug_model, str(debug_path))

    try:
        x = np.random.rand(1, 3, 640, 640).astype(np.float32)

        sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
        (out_orig,) = sess_orig.run([ORIGINAL_OUTPUT_NAME], {"images": x})

        sess_debug = ort.InferenceSession(str(debug_path), providers=["CPUExecutionProvider"])
        out_fused, boxes_val, scores_val = sess_debug.run(
            [ORIGINAL_OUTPUT_NAME, "nms_boxes", "nms_scores"], {"images": x}
        )
    finally:
        debug_path.unlink(missing_ok=True)

    if not np.array_equal(out_orig, out_fused):
        raise RuntimeError(
            f"sanity check FAILED: {ORIGINAL_OUTPUT_NAME} differs between the original and "
            "fused graphs -- the graph surgery must not have been purely additive. Not saving."
        )
    if boxes_val.shape != (1, 25200, 4):
        raise RuntimeError(f"sanity check FAILED: boxes shape {boxes_val.shape} != (1, 25200, 4)")
    if scores_val.shape != (1, 25200, 1):
        raise RuntimeError(f"sanity check FAILED: scores shape {scores_val.shape} != (1, 25200, 1)")
    if not np.isfinite(boxes_val).all() or not np.isfinite(scores_val).all():
        raise RuntimeError("sanity check FAILED: boxes/scores contain NaN/Inf")
    # NOTE: no value-range check on boxes -- random NOISE input (not a real
    # image) routinely sends this model's raw keypoint regression well
    # outside [0,640], since nothing in the architecture clamps it; that's
    # normal for random input and not a graph-surgery bug. Real-image
    # range/correctness is verified on-device against the known demo-video
    # armor-count baseline instead.
    print(
        f"sanity check passed: {ORIGINAL_OUTPUT_NAME} byte-identical pre/post-fusion, "
        f"boxes {boxes_val.shape} scores {scores_val.shape} (up through the EfficientNMS_TRT "
        "node's inputs -- the node itself has no onnxruntime kernel to test locally, its "
        "parse+build correctness is verified on-device on the Jetson instead)"
    )


def main():
    repo_root = Path(__file__).resolve().parents[2]
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "assets" / "yolov5.onnx"

    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)

    if any(o.name == "num_detections" for o in graph.outputs):
        raise RuntimeError(
            f"{onnx_path} already has a 'num_detections' output -- looks already fused with "
            "this script. Re-running would fuse a second EfficientNMS_TRT node on top. If you "
            "want to re-fuse from scratch, start from scripts/onnx/yolov5_prefusion_backup.onnx "
            "(or a fresh openvino2onnx export of assets/yolov5.xml) instead."
        )
    if any(o.name == "selected_indices" for o in graph.outputs):
        raise RuntimeError(
            f"{onnx_path} has a 'selected_indices' output -- that's the OLDER DDS-based "
            "fuse_nms.py fusion, not compatible with this script (it doesn't know how to "
            "strip that fusion back out first). Start from "
            "scripts/onnx/yolov5_prefusion_backup.onnx instead, e.g.:\n"
            "  python scripts/onnx/fuse_efficient_nms.py scripts/onnx/yolov5_prefusion_backup.onnx"
        )

    fuse(graph)

    # onnx.checker validates node op_types against known operator schemas --
    # EfficientNMS_TRT is a TensorRT-proprietary plugin op with no ONNX
    # schema (that's inherent to it being a plugin, not a spec gap here),
    # so full graph-wide checking isn't meaningful the way it was in
    # fuse_nms.py (whose NonMaxSuppression is a real ONNX op). Skip it for
    # this graph; sanity_check() below validates everything that CAN be
    # validated outside of TensorRT itself, and the actual parse+build is
    # verified on-device on the Jetson (the only tool that understands this
    # op at all is nvonnxparser + the plugin registry).
    sanity_check(onnx_path, graph)

    fused_model = gs.export_onnx(graph)
    tmp_path = onnx_path.with_suffix(".fused_tmp.onnx")
    onnx.save(fused_model, str(tmp_path))
    tmp_path.replace(onnx_path)
    print(f"saved {onnx_path} (outputs: {[o.name for o in graph.outputs]})")


if __name__ == "__main__":
    main()
