import CoreImage
import Vision

// MARK: - BusApproachTracker

/// Thin orchestrator. Wires BusDetectionFlow → BusTrackingFlow → BusInfoFlow
/// through WorkflowManager. Public API is identical to the former monolith.
public final class BusApproachTracker {

    // MARK: Public Types (unchanged from original API)

    public struct Config: Sendable {
        public var detectorW: Int = 512
        public var detectorH: Int = 896
        public var stage1BusClass: Int = 5
        public var stage1Conf: Double = 0.51
        public var stage2Conf: Double = 0.50
        public var preferLargestStage2Box: Bool = false
        public var iouMatchThreshold: Double = 0.2
        public var maxMissedFrames: Int = 40
        public var approachWindow: Int = 5
        public var approachMinFrames: Int = 2
        public var approachRatioThreshold: Double = 0.001
        public var stage1CropMargin: Double = 0.00
        public var stage2CropMargin: Double = 0.00
        public init() {}
    }

    public struct BusResult: Sendable {
        public let id: String
        public let ocrText: String
        public let bboxDetectorSpace: CGRect
        public let bboxOriginalTopLeft: CGRect
        public let confidence: Double
        public let approachingScore: Double
    }

    /// Timing information for each stage of the workflow.
    public struct TimingInfo: Sendable {
        public let flowTimings: [(name: String, ms: Double)]
        public let totalMs: Double
    }

    // MARK: Flows & engine

    private let detectionFlow: BusDetectionFlow
    private let trackingFlow: BusTrackingFlow
    private let infoFlow: BusInfoFlow
    private let stage1Origin: YOLOModel.BoxOrigin
    private let detectorH: Double

    // MARK: Init

    public init?(stage1Model: YOLOModel?, stage2Model: YOLOModel?, config: Config = .init()) {
        guard let s1 = stage1Model, let s2 = stage2Model else { return nil }

        stage1Origin = s1.boxOrigin
        detectorH = Double(config.detectorH)

        var d1cfg = BusDetector.Config()
        d1cfg.detectorW = config.detectorW; d1cfg.detectorH = config.detectorH
        d1cfg.busClass = config.stage1BusClass; d1cfg.confidence = config.stage1Conf
        d1cfg.cropMargin = config.stage1CropMargin

        var tcfg = BusTracker.Config()
        tcfg.iouMatchThreshold = config.iouMatchThreshold
        tcfg.maxMissedFrames = config.maxMissedFrames
        tcfg.approachWindow = config.approachWindow
        tcfg.approachMinFrames = config.approachMinFrames
        tcfg.approachRatioThreshold = config.approachRatioThreshold

        var d2cfg = BusInfoDetector.Config()
        d2cfg.detectorW = config.detectorW; d2cfg.detectorH = config.detectorH
        d2cfg.confidence = config.stage2Conf
        d2cfg.preferLargestBox = config.preferLargestStage2Box
        d2cfg.cropMargin = config.stage2CropMargin

        detectionFlow = BusDetectionFlow(detector: BusDetector(model: s1, config: d1cfg))
        trackingFlow  = BusTrackingFlow(tracker: BusTracker(config: tcfg))
        infoFlow      = BusInfoFlow(infoDetector: BusInfoDetector(model: s2, config: d2cfg))
    }

    // MARK: Public API

    public func processFrame(
        _ frame: CIImage,
        ocrPreset: OCRPreset = .default,
        recognitionLanguages: [String]? = ["pt-PT"],
        usesLanguageCorrection: Bool = false,
        recognitionLevel: VNRequestTextRecognitionLevel = .accurate
    ) async throws -> (results: [BusResult], timing: TimingInfo) {

        let workflowStart = CFAbsoluteTimeGetCurrent()
        var flowTimings: [(name: String, ms: Double)] = []

        // Stage 1 — detection
        var t0 = CFAbsoluteTimeGetCurrent()
        let detOut: BusDetectionFlow.Output
        do {
            detOut = try await detectionFlow.run(input: frame)
        } catch {
            let totalMs = (CFAbsoluteTimeGetCurrent() - workflowStart) * 1000
            return ([], TimingInfo(flowTimings: flowTimings, totalMs: totalMs))
        }
        flowTimings.append((detectionFlow.id, (CFAbsoluteTimeGetCurrent() - t0) * 1000))

        // Stage 2 — tracking
        t0 = CFAbsoluteTimeGetCurrent()
        let tracked = try await trackingFlow.run(input: detOut.detections)
        flowTimings.append((trackingFlow.id, (CFAbsoluteTimeGetCurrent() - t0) * 1000))

        let approaching = tracked.filter { $0.isApproaching }
        guard !approaching.isEmpty else {
            let totalMs = (CFAbsoluteTimeGetCurrent() - workflowStart) * 1000
            return ([], TimingInfo(flowTimings: flowTimings, totalMs: totalMs))
        }

        // Stage 3 — info detection per approaching bus
        t0 = CFAbsoluteTimeGetCurrent()
        let originalCI = frame
        let originalW = Double(frame.extent.width)
        let originalH = Double(frame.extent.height)

        var results: [BusResult] = []

        for bus in approaching {
            let busBoxOrig: Box
            if let match = detOut.detections.first(where: { iou($0.boxDetector, bus.lastBoxDetector) >= 0.5 }) {
                busBoxOrig = match.boxOriginal
            } else {
                continue
            }
            if busBoxOrig.w < 2 || busBoxOrig.h < 2 { continue }

            let infoInput = BusInfoFlow.Input(
                bus: bus, originalCI: originalCI,
                originalW: originalW, originalH: originalH,
                ocrPreset: ocrPreset,
                recognitionLanguages: recognitionLanguages,
                usesLanguageCorrection: usesLanguageCorrection,
                recognitionLevel: recognitionLevel
            )

            let infoResult = try await infoFlow.run(input: infoInput)

            let detectorRect = boxFromTopLeft(bus.lastBoxDetector, to: stage1Origin,
                                              inputH: detectorH).rectXYWH()
            results.append(BusResult(
                id: bus.name, ocrText: infoResult.ocrText,
                bboxDetectorSpace: detectorRect,
                bboxOriginalTopLeft: CGRect(x: busBoxOrig.x1, y: busBoxOrig.y1,
                                           width: busBoxOrig.w, height: busBoxOrig.h),
                confidence: bus.lastScore, approachingScore: bus.approachingScore
            ))
        }
        flowTimings.append((infoFlow.id, (CFAbsoluteTimeGetCurrent() - t0) * 1000))

        results.sort { $0.id < $1.id }
        let totalMs = (CFAbsoluteTimeGetCurrent() - workflowStart) * 1000
        return (results, TimingInfo(flowTimings: flowTimings, totalMs: totalMs))
    }

    // MARK: - Helpers

    private func iou(_ a: Box, _ b: Box) -> Double {
        let xA = max(a.x1, b.x1), yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2), yB = min(a.y2, b.y2)
        let inter = max(0, xB - xA) * max(0, yB - yA)
        let union = a.area + b.area - inter
        if union <= 0 { return 0 }
        return inter / union
    }

    private func boxFromTopLeft(_ b: Box, to origin: YOLOModel.BoxOrigin, inputH: Double) -> Box {
        guard origin == .bottomLeft else { return b }
        return Box(x1: b.x1, y1: inputH - b.y2, x2: b.x2, y2: inputH - b.y1)
    }
}



