import CoreImage
import CoreGraphics
import Vision

// MARK: - BusInfoFlow

/// Thin adapter: wraps BusInfoDetector, conforming to the Flow protocol.
/// Input: a tracked bus + the original frame CIImage. Output: BusInfoResult.
public struct BusInfoFlow: Flow {

    public struct Input: Sendable {
        public let bus: TrackedBus
        public let originalCI: CIImage
        public let originalW: Double
        public let originalH: Double
        public let ocrPreset: OCRPreset
        public let recognitionLanguages: [String]?
        public let usesLanguageCorrection: Bool
        public let recognitionLevel: VNRequestTextRecognitionLevel

        public init(bus: TrackedBus, originalCI: CIImage,
                    originalW: Double, originalH: Double,
                    ocrPreset: OCRPreset,
                    recognitionLanguages: [String]? = nil,
                    usesLanguageCorrection: Bool = false,
                    recognitionLevel: VNRequestTextRecognitionLevel = .accurate) {
            self.bus = bus; self.originalCI = originalCI
            self.originalW = originalW; self.originalH = originalH
            self.ocrPreset = ocrPreset
            self.recognitionLanguages = recognitionLanguages
            self.usesLanguageCorrection = usesLanguageCorrection
            self.recognitionLevel = recognitionLevel
        }
    }

    public let id: String
    private let infoDetector: BusInfoDetector

    public init(id: String = "BusInfo", infoDetector: BusInfoDetector) {
        self.id = id
        self.infoDetector = infoDetector
    }

    public func run(input: Input) async throws -> BusInfoResult {
        let busCropCI = cropFromTopLeftAndNormalize(
            input.originalCI,
            srcW: input.originalW, srcH: input.originalH,
            boxTopLeft: input.bus.lastBoxDetector
        )
        let busCropW = Double(busCropCI.extent.width)
        let busCropH = Double(busCropCI.extent.height)

        guard busCropW >= 2, busCropH >= 2 else {
            return BusInfoResult(ocrText: "", infoBoxOriginal: nil)
        }

        return try await infoDetector.detect(
            busCropCI: busCropCI, busCropW: busCropW, busCropH: busCropH,
            ocrPreset: input.ocrPreset,
            recognitionLanguages: input.recognitionLanguages,
            usesLanguageCorrection: input.usesLanguageCorrection,
            recognitionLevel: input.recognitionLevel
        )
    }

    private func cropFromTopLeftAndNormalize(_ src: CIImage, srcW: Double, srcH: Double, boxTopLeft: Box) -> CIImage {
        let ciY = srcH - (boxTopLeft.y1 + boxTopLeft.h)
        let cropRect = CGRect(x: boxTopLeft.x1, y: ciY, width: boxTopLeft.w, height: boxTopLeft.h).integral
        let bounds = CGRect(x: 0, y: 0, width: srcW, height: srcH)
        let rr = cropRect.intersection(bounds).integral
        if rr.isNull || rr.width < 1 || rr.height < 1 { return CIImage.empty() }
        let cropped = src.cropped(to: rr)
        return cropped.transformed(by: CGAffineTransform(translationX: -rr.origin.x, y: -rr.origin.y))
    }
}
