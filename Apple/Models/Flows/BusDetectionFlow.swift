import CoreImage

// MARK: - BusDetectionFlow

/// Thin adapter: wraps BusDetector, conforming to the Flow protocol.
/// Input: CIImage (full frame). Output: detections + letterbox metadata.
public struct BusDetectionFlow: Flow {

    public struct Output: Sendable {
        public let detections: [BusDetection]
        public let meta: LetterboxMeta
    }

    public let id: String
    private let detector: BusDetector

    public init(id: String = "BusDetection", detector: BusDetector) {
        self.id = id
        self.detector = detector
    }

    public func run(input: CIImage) async throws -> Output {
        let (detections, meta) = try detector.detect(frame: input)
        return Output(detections: detections, meta: meta)
    }
}
