# Architecture Intent

## WorkflowEngine Refactor

Refactoring `BusAndOCR.swift` into a modular workflow engine.

---

## Target Structure

```
WorkflowEngine/
  - Flow.swift
  - WorkflowGraph.swift
  - WorkflowManager.swift
  - WorkflowOutput.swift

Flows/
  - BusDetectionFlow.swift
  - BusTrackingFlow.swift
  - BusInfoFlow.swift

Domain/
  - YOLOModel.swift
  - ImageLetterboxer.swift
  - OCRPipeline.swift        (owns OCRPreset)
  - BusDetector.swift
  - BusTracker.swift
  - BusInfoDetector.swift
```

---

## Rules

- Flows never import each other
- WorkflowManager has zero domain knowledge
- BusTracker has zero image framework imports
- OCRPreset lives in OCRPipeline.swift only

---

## Commit Conventions

- Title: 50 chars max, imperative mood, no period
- Description: what changed, why, and what was deliberately
  left unchanged (e.g. "BusAndOCR.swift kept compiling via
  internal delegation")
- Never commit if project does not build

---

## Step 1 — Extract leaf dependencies

```
Extract YOLOModel, ImageLetterboxer, and OCRPipeline
into separate files per CLAUDE.md. Move parseDetections() and
detectionToTopLeft() into YOLOModel. OCRPreset moves into
OCRPipeline.swift. BusAndOCR.swift should still compile
with internal typealias bridges until refactor is complete.

After confirming the project builds, commit with:
  Title: Extract leaf domain types into separate files
  Description:
  - YOLOModel.swift: owns model loading, prediction,
    parseDetections(), detectionToTopLeft(), YOLODetection type
  - ImageLetterboxer.swift: owns letterbox transform,
    LetterboxMeta, pixel buffer helpers, CIImage rendering
  - OCRPipeline.swift: owns OCRPreset, all preprocessing
    steps, thinning, Vision OCR request
  - BusAndOCR.swift: kept compiling via typealiases,
    no logic changes, no public API changes
```

---

## Step 2 — Extract domain flow types

```
Create BusDetector.swift, BusTracker.swift, and
BusInfoDetector.swift per CLAUDE.md. BusTracker must have
zero CoreImage/Vision imports — pure logic only. Delegate
to new types internally from BusAndOCR.swift so it still
compiles and passes existing call sites unchanged.

After confirming the project builds, commit with:
  Title: Extract domain flow types from monolith
  Description:
  - BusDetector.swift: Stage1 inference, box unprojection,
    confidence filtering, returns BusDetection with both
    detector-space and original-space boxes
  - BusTracker.swift: IoU matching, track lifecycle,
    approach scoring — zero image framework imports
  - BusInfoDetector.swift: Stage2 inference, best-box
    selection, info crop, delegates to OCRPipeline
  - BusAndOCR.swift: now delegates to above types,
    public API and behavior unchanged
```

---

## Step 3 — Build the WorkflowEngine

```
Create the WorkflowEngine layer per CLAUDE.md:
Flow protocol, WorkflowGraph, WorkflowManager, WorkflowOutput.
WorkflowManager must have zero domain knowledge — no bus types,
no CoreML, no Vision imports. Flows with independent inputs
run concurrently; serial flows receive typed upstream output.
Do not wire any existing flows yet.

After confirming the project builds, commit with:
  Title: Add generic WorkflowEngine layer
  Description:
  - Flow.swift: Flow protocol with typed Input/Output,
    FlowStatus enum (success/failed/skipped), FlowResult
  - WorkflowGraph.swift: topology declaration, concurrent
    groups and serial dependencies, no domain knowledge
  - WorkflowManager.swift: async execution engine,
    runs concurrent groups in parallel via TaskGroup,
    passes typed upstream output to serial dependents
  - WorkflowOutput.swift: collected results keyed by
    flowId, type-safe result retrieval
  - No existing files modified
```

---

## Step 4 — Add Flow adapter layer

```
Create BusDetectionFlow, BusTrackingFlow, BusInfoFlow
in the Flows/ directory per CLAUDE.md. Each is a thin adapter
conforming to the Flow protocol and wrapping the corresponding
Domain type. Flows must not import each other. Do not modify
any Domain or WorkflowEngine files.

After confirming the project builds, commit with:
  Title: Add Flow adapters for bus processing pipeline
  Description:
  - BusDetectionFlow.swift: wraps BusDetector, Input is
    CGImage, Output is [BusDetection] + LetterboxMeta
  - BusTrackingFlow.swift: wraps BusTracker, Input is
    [BusDetection], Output is [TrackedBus]
  - BusInfoFlow.swift: wraps BusInfoDetector, Input is
    TrackedBus + CGImage, Output is BusResult
  - Flows are stateless adapters only, all state lives
    in Domain types
  - No Domain or WorkflowEngine files modified
```

---

## Step 5 — Replace monolith with orchestrator

```
Replace BusAndOCR.swift with a thin BusApproachTracker
that wires BusDetectionFlow, BusTrackingFlow, and BusInfoFlow
through WorkflowManager per CLAUDE.md. Public API of
BusApproachTracker must remain identical so call sites need
no changes. Once confirmed building and behavior equivalent,
delete BusAndOCR.swift.

After confirming the project builds, commit with:
  Title: Replace monolith with WorkflowManager orchestrator
  Description:
  - BusApproachTracker.swift: thin orchestrator, wires
    the 3 bus flows through WorkflowManager, processFrame()
    public signature unchanged
  - BusDetectionFlow and SegmentationFlow (future) can now
    run concurrently by adding to the same concurrent group
  - BusAndOCR.swift deleted — all logic now lives in
    Domain/, Flows/, and WorkflowEngine/
  - No call site changes required
