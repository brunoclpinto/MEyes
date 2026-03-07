import CoreImage
import CoreGraphics
import Vision

// MARK: - BusInfoResult

public struct BusInfoResult: Sendable {
    public let ocrText: String
    /// Info box in original frame coordinates (top-left origin), or nil if none found.
    public let infoBoxOriginal: Box?

    public init(ocrText: String, infoBoxOriginal: Box?) {
        self.ocrText = ocrText
        self.infoBoxOriginal = infoBoxOriginal
    }
}

// MARK: - BusInfoDetector

/// Stage2: given a bus crop from the original frame, runs YOLO to find the info region,
/// crops it, and runs OCR. Delegates all image preprocessing to OCRPipeline.
public final class BusInfoDetector {

    public struct Config: Sendable {
        public var detectorW: Int = 512
        public var detectorH: Int = 896
        public var confidence: Double = 0.50
        public var preferLargestBox: Bool = false
        public var cropMargin: Double = 0.00
        public init() {}
    }

    public let model: YOLOModel
    public var config: Config

    public init(model: YOLOModel, config: Config = .init()) {
        self.model = model
        self.config = config
    }

    /// Run Stage2 on a pre-cropped bus image (origin already normalized to 0,0).
    public func detect(
        busCropCI: CIImage,
        busCropW: Double,
        busCropH: Double,
        ocrPreset: OCRPreset,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) async throws -> BusInfoResult {

        let dstW = Double(config.detectorW)
        let dstH = Double(config.detectorH)

        let (letterboxed, meta) = ImageLetterboxer.letterboxWithMeta(
            busCropCI, srcW: busCropW, srcH: busCropH, dstW: dstW, dstH: dstH
        )

        let pb = try ImageLetterboxer.makePixelBuffer(width: config.detectorW, height: config.detectorH)
        ImageLetterboxer.render(letterboxed, to: pb)

        let raw = try model.predict(pixelBuffer: pb)
        let parsed = YOLOModel.parseDetections(raw, inputW: dstW, inputH: dstH)
            .map { YOLOModel.detectionToTopLeft($0, origin: model.boxOrigin, inputH: dstH) }
            .filter { $0.score >= config.confidence }

        guard let best = pickBest(parsed) else {
            return BusInfoResult(ocrText: "", infoBoxOriginal: nil)
        }

        var infoBox = Box(x1: best.x1, y1: best.y1, x2: best.x2, y2: best.y2)
        infoBox = meta.dstToSrc(infoBox)
        infoBox = meta.clampToSrc(infoBox)
        infoBox = expandBox(infoBox, srcW: busCropW, srcH: busCropH, margin: config.cropMargin)

        if infoBox.w < 2 || infoBox.h < 2 {
            return BusInfoResult(ocrText: "", infoBoxOriginal: infoBox)
        }

        let infoCropCI = cropFromTopLeftAndNormalize(
            busCropCI, srcW: busCropW, srcH: busCropH, boxTopLeft: infoBox
        )

        guard infoCropCI.extent.width >= 2, infoCropCI.extent.height >= 2 else {
            return BusInfoResult(ocrText: "", infoBoxOriginal: infoBox)
        }

        let ocrText = try await OCRPipeline.ocrCIImage(
            infoCropCI, preset: ocrPreset,
            recognitionLanguages: recognitionLanguages,
            usesLanguageCorrection: usesLanguageCorrection,
            recognitionLevel: recognitionLevel
        )

        return BusInfoResult(ocrText: ocrText, infoBoxOriginal: infoBox)
    }

    // MARK: - Helpers

    private func pickBest(_ detections: [YOLODetection]) -> YOLODetection? {
        guard !detections.isEmpty else { return nil }
        if config.preferLargestBox {
            return detections.max(by: { $0.area < $1.area })
        } else {
            return detections.max(by: { $0.score < $1.score })
        }
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
