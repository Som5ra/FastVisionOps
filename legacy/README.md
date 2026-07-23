# Legacy adapters

These directories preserve the original standalone scripts and call
signatures for reference. They delegate to the tested `fastvisionops` or
`nmss` packages and are not installed.

| Original project area | Adapter |
| --- | --- |
| NumPy bounding-box NMS | `bbox-nms/` |
| Native and batched bounding-box NMS | `bbox-nms-c-version/` |
| Boolean-mask NMS | `mask-nms/` |

New code should use the public `fastvisionops` API documented in the
[root README](../README.md).
