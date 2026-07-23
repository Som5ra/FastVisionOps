# Legacy mask NMS API

This directory preserves the original mask NMS function names. The maintained
implementation is `nmss.mask` and requires only NumPy.

```python
from fastvisionops import mask_nms, multiclass_mask_nms
```

Inputs use shape `(num_masks, ...)`, boolean dtype, and scores shaped
`(num_masks,)` or `(num_masks, num_classes)`. See the
[root README](../README.md) for examples and semantics.
