# Setting Up Cosmos3 with Isaac Sim Replicator While Preserving Object Identity

This note describes how to generate Isaac Sim Replicator outputs for NVIDIA Cosmos so that object identities are preserved across frames and available to downstream processing.[cite:13][cite:15]

## Overview

The standard integration path uses Isaac Sim Replicator's `CosmosWriter`, which exports multi-modal clip data such as RGB, depth, segmentation, edges, and shaded segmentation in a layout intended for Cosmos synthetic data workflows.[cite:13]

Identity preservation is achieved primarily through the segmentation channel: when `CosmosWriter` is initialized with `use_instance_id=True`, each object instance is represented by a stable instance ID rather than only a class label.[cite:13]

## Identity-preserving configuration

A minimal Replicator setup attaches `CosmosWriter` to a render product and enables instance-ID output so that the segmentation stream carries per-object identity information across frames.[cite:13]

```python
import os
import omni.replicator.core as rep
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

render_product = rep.create.render_product("/World/Camera", (1280, 720))

backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=os.path.join(os.getcwd(), "_out_cosmos_scene"))

writer = rep.WriterRegistry.get("CosmosWriter")
writer.initialize(
    backend=backend,
    use_instance_id=True,
)
writer.attach(render_product)

for _ in range(100):
    rep.orchestrator.step(pause_timeline=False)

rep.orchestrator.wait_until_complete()
writer.detach()
render_product.destroy()
simulation_app.close()
```

This mode is preferable when the goal is to preserve individual objects, such as distinguishing `Rack_01` from `Rack_02`, rather than only preserving semantic categories like `rack`.[cite:13]

## What metadata reaches Cosmos

In the documented workflow, Cosmos consumes the generated control modalities as files rather than a separate required metadata API payload, and the identity signal is embedded in the segmentation outputs produced by Replicator.[cite:13][cite:15]

When using `use_instance_id=True`, the segmentation PNG frames and `segmentation.mp4` encode object-level IDs; when using `segmentation_mapping`, the output instead encodes semantic class mappings, which preserves category identity but not unique per-instance identity.[cite:13]

That distinction matters operationally: instance-ID mode supports object-wise continuity across time, while semantic mapping supports class-consistent augmentation without uniquely tracking individual scene instances.[cite:13][cite:15]

## Output structure

The Isaac Sim Cosmos tutorial documents a clip-based directory structure in which each clip contains modality subfolders and rendered MP4 files for the same sequence.[cite:13]

A representative layout is shown below.[cite:13]

```text
_out_cosmos_scene/
  clip_0000/
    rgb/
    depth/
    segmentation/
    shaded_seg/
    edges/
    rgb.mp4
    depth.mp4
    segmentation.mp4
    shaded_seg.mp4
    edges.mp4
```

Cosmos workflows then use these modality files as control inputs, for example combining RGB with segmentation, edges, and depth to preserve scene geometry and identity structure while changing appearance or environmental style.[cite:13][cite:15]

## Recommended sidecar metadata

If additional metadata is needed beyond the built-in segmentation identity signal, the practical pattern is to keep a sidecar mapping from Replicator instance IDs to scene metadata such as USD prim path, semantic class, or custom attributes.[cite:13]

A typical sidecar file can look like this:

```json
{
  "instances": {
    "13": {
      "prim_path": "/World/Rack_01",
      "class": "rack"
    },
    "42": {
      "prim_path": "/World/Robot_01",
      "class": "robot"
    }
  }
}
```

This design keeps the pixel-aligned identity inside the segmentation channel while allowing downstream code to join each instance ID with richer symbolic metadata for analytics, filtering, or evaluation.[cite:13]

## Practical constraints

Identity stability depends on maintaining stable scene prims across frames; if objects are deleted and recreated, new instance IDs may be assigned and continuity can break even when instance-ID output is enabled.[cite:13]

For that reason, long synthetic clips intended for tracking, video world modeling, or consistent object augmentation should avoid scene regeneration patterns that remint prim identities unless the downstream pipeline explicitly remaps them.[cite:13][cite:15]
