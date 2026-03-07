import CoreImage

// MARK: - BusDetection

/// Result of Stage1 inference for a single bus candidate.
public struct BusDetection: Sendable {
    /// Bounding box in detector-input space (top-left origin).
    public let boxDetector: Box
    /// Bounding box mapped back to the original frame (top-left origin).
    public let boxOriginal: Box
    public let score: Double
    public let cls: Int

    public init(boxDetector: Box, boxOriginal: Box, score: Double, cls: Int) {
        self.boxDetector = boxDetector
        self.boxOriginal = boxOriginal
        self.score = score
        self.cls = cls
    }
}

// MARK: - BusDetector

/// Stage1: runs YOLO on a letterboxed frame, filters bus detections,
/// and unprojects boxes back to original pixel space.
public final class BusDetector {

    public struct Config: Sendable {
        public var detectorW: Int = 512
        public var detectorH: Int = 896
        public var busClass: Int = 5
        public var confidence: Double = 0.51
        public var cropMargin: Double = 0.00
        public init() {}
    }

    public let model: YOLOModel
    public var config: Config

    public init(model: YOLOModel, config: Config = .init()) {
        self.model = model
        self.config = config
    }

    /// Run Stage1 on `frame`. Returns detected buses sorted by score descending.
    public func detect(frame: CIImage) throws -> (detections: [BusDetection], meta: LetterboxMeta) {
        let srcW = Double(frame.extent.width)
        let srcH = Double(frame.extent.height)
        let dstW = Double(config.detectorW)
        let dstH = Double(config.detectorH)

        let originalCI = frame
        let (letterboxed, meta) = ImageLetterboxer.letterboxWithMeta(
            originalCI, srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH
        )

        let pb = try ImageLetterboxer.makePixelBuffer(width: config.detectorW, height: config.detectorH)
        ImageLetterboxer.render(letterboxed, to: pb)

        let raw = try model.predict(pixelBuffer: pb)
        let parsed = YOLOModel.parseDetections(raw, inputW: dstW, inputH: dstH)
            .map { YOLOModel.detectionToTopLeft($0, origin: model.boxOrigin, inputH: dstH) }
            .filter { $0.cls == config.busClass && $0.score >= config.confidence }
            .sorted { $0.score > $1.score }

        let detections: [BusDetection] = parsed.map { d in
            let detBox = Box(x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2)
            var origBox = meta.dstToSrc(detBox)
            origBox = meta.clampToSrc(origBox)
            origBox = expandBox(origBox, srcW: srcW, srcH: srcH, margin: config.cropMargin)
            return BusDetection(boxDetector: detBox, boxOriginal: origBox, score: d.score, cls: d.cls)
        }

        return (detections, meta)
    }

    private func expandBox(_ b: Box, srcW: Double, srcH: Double, margin: Double) -> Box {
        guard margin > 0 else { return clampBox(b, srcW: srcW, srcH: srcH) }
        let mx = b.w * margin, my = b.h * margin
        return clampBox(Box(x1: b.x1 - mx, y1: b.y1 - my,
                            x2: b.x2 + mx, y2: b.y2 + my), srcW: srcW, srcH: srcH)
    }

    private func clampBox(_ b: Box, srcW: Double, srcH: Double) -> Box {
        let x1 = max(0, min(srcW, b.x1)), x2 = max(0, min(srcW, b.x2))
        let y1 = max(0, min(srcH, b.y1)), y2 = max(0, min(srcH, b.y2))
        return Box(x1: min(x1, x2), y1: min(y1, y2), x2: max(x1, x2), y2: max(y1, y2))
    }
}
