//
//  BusApproachTracker_HiResCrops.swift
//
//  What changed vs the previous Xcode version:
//  ✅ Stage 1 + Stage 2 models still run at a fixed detector size (default 536×720)
//  ✅ BUT all crops (bus crop + info crop) are taken from the ORIGINAL high-res frame
//     by un-letterboxing boxes back into the original coordinate space.
//  ✅ OCR runs on the highest-res possible crop (info region cropped from original pixels),
//     then your preprocessing (scale/flatten/core/binarize/…) is applied.
//
//  Model assumption:
//   - CoreML output is MLMultiArray shaped N×6 or 1×N×6 with rows:
//       [x1, y1, x2, y2, score, class]
//   - Coordinates are either pixel coords in model-input space, or normalized (0..1-ish).
//
//  Usage (Xcode):
//    let s1 = try YOLOModel(.bundle(name: "yolo26sINT8"), boxOrigin: .topLeft)
//    let s2 = try YOLOModel(.bundle(name: "busInfoYolo26sINT8"), boxOrigin: .topLeft)
//    var cfg = BusApproachTracker.Config()
//    cfg.detectorW = 536
//    cfg.detectorH = 720
//    cfg.stage1BusClass = 5
//    cfg.stage1Conf = 0.51
//    cfg.stage2Conf = 0.51
//    let tracker = BusApproachTracker(stage1Model: s1, stage2Model: s2, config: cfg)
//
//    let preset = OCRPreset(scale: 1, flatten: true, core: 1, binarize: false)
//    let approaching = try await tracker.processFrame(cgImage, ocrPreset: preset)
//
//    approaching.forEach { print($0.id, $0.ocrText) }
//

import Foundation
import CoreML
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import Metal
import CoreGraphics

// MARK: - OCR Preset

public struct OCRPreset: Sendable {
    public enum Thin: Sendable {
        case disabled
        /// maxIterations: 0 = until stable
        case enabled(maxIterations: Int = 0)
    }

    public var scale: Float
    public var flatten: Bool
    public var core: Float
    public var binarize: Bool
    public var thresh: Float?     // 0..1, nil => Otsu when binarize=true
    public var shrink: Float
    public var thin: Thin
    public var regrow: Float

    public init(
        scale: Float = 1,
        flatten: Bool = true,
        core: Float = 1,
        binarize: Bool = false,
        thresh: Float? = nil,
        shrink: Float = 0,
        thin: Thin = .disabled,
        regrow: Float = 0
    ) {
        self.scale = scale
        self.flatten = flatten
        self.core = core
        self.binarize = binarize
        self.thresh = thresh
        self.shrink = shrink
        self.thin = thin
        self.regrow = regrow
    }

    /// “Preprocessing disabled”: minimal grayscale and then Vision.
    public static let disabled = OCRPreset(
        scale: 1,
        flatten: false,
        core: 0,
        binarize: false,
        thresh: nil,
        shrink: 0,
        thin: .disabled,
        regrow: 0
    )
}

// MARK: - YOLOModel wrapper

public final class YOLOModel: @unchecked Sendable {
    public enum Source {
        case bundle(name: String, bundle: Bundle = .main) // expects <name>.mlmodelc at runtime
        case mlModel(MLModel)
        case compiledURL(URL) // .mlmodelc
    }

    /// How to interpret YOLO y-coordinates (for cropping).
    public enum BoxOrigin: Sendable {
        case topLeft
        case bottomLeft
    }

    public let model: MLModel
    public let inputName: String
    public let outputName: String
    public let boxOrigin: BoxOrigin

    public init(_ source: Source, boxOrigin: BoxOrigin = .topLeft, computeUnits: MLComputeUnits = .all) throws {
        self.boxOrigin = boxOrigin

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        switch source {
        case .bundle(let name, let bundle):
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
                throw NSError(domain: "YOLOModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(name).mlmodelc in bundle"])
            }
            self.model = try MLModel(contentsOf: url, configuration: cfg)

        case .mlModel(let m):
            self.model = m

        case .compiledURL(let url):
            self.model = try MLModel(contentsOf: url, configuration: cfg)
        }

        let inputs = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard !inputs.isEmpty else {
            throw NSError(domain: "YOLOModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model has no inputs"])
        }
        self.inputName = inputs.contains("image") ? "image" : inputs[0]

        let outputs = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard !outputs.isEmpty else {
            throw NSError(domain: "YOLOModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Model has no outputs"])
        }
        self.outputName = outputs[0]
    }

    public func predict(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let out = try model.prediction(from: provider)
        guard let fv = out.featureValue(for: outputName), let arr = fv.multiArrayValue else {
            throw NSError(domain: "YOLOModel", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output \(outputName) is not MLMultiArray"])
        }
        return arr
    }
}

// MARK: - BusApproachTracker (Hi-Res crops)

public final class BusApproachTracker {

    // MARK: Public Types

    public struct Config: Sendable {
        /// Detector input size (Stage1 & Stage2 input)
        public var detectorW: Int = 512
        public var detectorH: Int = 896

        /// Stage 1 (bus detect)
        public var stage1BusClass: Int = 5
        public var stage1Conf: Double = 0.51

        /// Stage 2 (info detect)
        public var stage2Conf: Double = 0.50
        public var preferLargestStage2Box: Bool = false

        /// Tracking
        public var iouMatchThreshold: Double = 0.2
        public var maxMissedFrames: Int = 40

        /// “Approaching” logic (area growth)
        public var approachWindow: Int = 5
        public var approachMinFrames: Int = 2
        public var approachRatioThreshold: Double = 0.001

        /// Optional: enlarge crop boxes (helps when boxes jitter).
        /// Expressed as fraction of box size (e.g. 0.10 = +10% each side).
        public var stage1CropMargin: Double = 0.00
        public var stage2CropMargin: Double = 0.00

        public init() {}
    }

    public struct BusResult: Sendable {
        public let id: String               // "Bus1", "Bus2", ...
        public let ocrText: String          // RAW OCR text (unfiltered), may be ""
        public let bboxDetectorSpace: CGRect // last Stage1 bbox in detector space (x,y,w,h) w/ YOLO origin
        public let bboxOriginalTopLeft: CGRect // mapped bbox in ORIGINAL top-left pixel coords
        public let confidence: Double
        public let approachingScore: Double
    }

    // MARK: Private Types

    private struct Detection {
        var x1: Double
        var y1: Double
        var x2: Double
        var y2: Double
        var score: Double
        var cls: Int
        var area: Double { max(0, x2 - x1) * max(0, y2 - y1) }
    }

    private struct Box {
        var x1: Double
        var y1: Double
        var x2: Double
        var y2: Double

        var w: Double { max(0, x2 - x1) }
        var h: Double { max(0, y2 - y1) }
        var area: Double { w * h }

        func rectXYWH() -> CGRect {
            CGRect(x: x1, y: y1, width: w, height: h)
        }
    }

    private struct Track {
        let numericId: Int
        let name: String
        var lastBoxDetector: Box
        var lastScore: Double
        var lastSeenFrame: Int
        var ageFrames: Int
        var areaHistory: [Double]      // in detector space (area ratios are stable)
        var approachingScore: Double
        var isApproaching: Bool
    }

    /// Letterbox metadata for mapping between original ↔ detector coordinate spaces (top-left origin).
    private struct LetterboxMeta {
        let srcW: Double
        let srcH: Double
        let dstW: Double
        let dstH: Double
        let scale: Double
        let padX: Double
        let padY: Double

        /// Map a box from dst (detector input) space back to src (original/crop) space.
        /// Coordinates are in pixel units with TOP-LEFT origin.
        func dstToSrc(_ b: Box) -> Box {
            let inv = 1.0 / max(scale, 1e-9)
            let x1 = (b.x1 - padX) * inv
            let x2 = (b.x2 - padX) * inv
            let y1 = (b.y1 - padY) * inv
            let y2 = (b.y2 - padY) * inv
            return Box(x1: x1, y1: y1, x2: x2, y2: y2)
        }

        func clampToSrc(_ b: Box) -> Box {
            let x1 = max(0, min(srcW, b.x1))
            let x2 = max(0, min(srcW, b.x2))
            let y1 = max(0, min(srcH, b.y1))
            let y2 = max(0, min(srcH, b.y2))
            return Box(x1: min(x1,x2), y1: min(y1,y2), x2: max(x1,x2), y2: max(y1,y2))
        }
    }

    // MARK: Stored Properties

    private let stage1: YOLOModel
    private let stage2: YOLOModel
    public var config: Config

    private var nextId: Int = 1
    private var tracks: [Int: Track] = [:]
    private var frameIndex: Int = 0

    // Shared CI context
    private static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    // MARK: Init

    public init?(stage1Model: YOLOModel?, stage2Model: YOLOModel?, config: Config = .init()) {
      guard
        let stage1 = stage1Model,
        let stage2 = stage2Model else {
        return nil
      }
      
        self.stage1 = stage1
        self.stage2 = stage2
        self.config = config
    }

    // MARK: Public API

    /// Feed one frame (CGImage). Returns buses that are approaching + OCR text.
    /// Crops are made from the ORIGINAL high-res frame.
    public func processFrame(
        _ frame: CGImage,
        ocrPreset: OCRPreset = OCRPreset(scale: 1, flatten: true, core: 1, binarize: false),
        recognitionLanguages: [String]? = nil,
        usesLanguageCorrection: Bool = false,
        recognitionLevel: VNRequestTextRecognitionLevel = .accurate
    ) async throws -> [BusResult] {

        frameIndex += 1

        let originalW = Double(frame.width)
        let originalH = Double(frame.height)

        // Original CIImage (extent origin is typically (0,0); still safe)
        let originalCI = CIImage(cgImage: frame)

        // 1) Letterbox ORIGINAL -> detector input size for Stage 1
        let (stage1InputCI, meta1) = Self.letterboxWithMeta(
            originalCI,
            srcW: originalW, srcH: originalH,
            dstW: Double(config.detectorW), dstH: Double(config.detectorH)
        )

        // 2) Stage 1 YOLO inference (on detector-sized image)
        let pb1 = try Self.makePixelBuffer(width: config.detectorW, height: config.detectorH)
        Self.render(stage1InputCI, to: pb1)

        let out1 = try stage1.predict(pixelBuffer: pb1)
        var det1 = Self.parseDetections(out1, inputW: Double(config.detectorW), inputH: Double(config.detectorH))
        det1 = det1
            .filter { $0.cls == config.stage1BusClass && $0.score >= config.stage1Conf }
            .sorted { $0.score > $1.score }

        // 3) Update tracks using DETECTOR space boxes (stable + fast)
        let matchedTrackIds = matchAndUpdateTracks(detections: det1)

        // 4) Remove stale tracks
        pruneStaleTracks()

        // 5) For approaching buses, do Stage 2 + OCR using ORIGINAL high-res crops
        var results: [BusResult] = []
        results.reserveCapacity(matchedTrackIds.count)

        for tid in matchedTrackIds.sorted() {
            guard let t = tracks[tid], t.isApproaching else { continue }

            // --- Map Stage1 bus box from detector space -> ORIGINAL space (top-left origin)
            var busBoxOrig = meta1.dstToSrc(t.lastBoxDetector)
            busBoxOrig = meta1.clampToSrc(busBoxOrig)
            busBoxOrig = expandBox(busBoxOrig, srcW: originalW, srcH: originalH, margin: config.stage1CropMargin)

            if busBoxOrig.w < 2 || busBoxOrig.h < 2 { continue }

            // --- Crop ORIGINAL high-res bus ROI, normalize origin to (0,0) for easier math downstream
            let busCropOriginalCI = cropFromTopLeftAndNormalize(
                originalCI,
                srcW: originalW, srcH: originalH,
                boxTopLeft: busBoxOrig
            )

            let busCropW = Double(busCropOriginalCI.extent.width)
            let busCropH = Double(busCropOriginalCI.extent.height)
            if busCropW < 2 || busCropH < 2 { continue }

            // --- Stage2 input: letterbox bus-crop -> detector input (still fixed size)
            let (stage2InputCI, meta2) = Self.letterboxWithMeta(
                busCropOriginalCI,
                srcW: busCropW, srcH: busCropH,
                dstW: Double(config.detectorW), dstH: Double(config.detectorH)
            )

            let pb2 = try Self.makePixelBuffer(width: config.detectorW, height: config.detectorH)
            Self.render(stage2InputCI, to: pb2)

            let out2 = try stage2.predict(pixelBuffer: pb2)
            var det2 = Self.parseDetections(out2, inputW: Double(config.detectorW), inputH: Double(config.detectorH))
            det2 = det2.filter { $0.score >= config.stage2Conf }

            guard let best2 = pickBestStage2(det2) else {
                results.append(BusResult(
                    id: t.name,
                    ocrText: "",
                    bboxDetectorSpace: t.lastBoxDetector.rectXYWH(),
                    bboxOriginalTopLeft: CGRect(x: busBoxOrig.x1, y: busBoxOrig.y1, width: busBoxOrig.w, height: busBoxOrig.h),
                    confidence: t.lastScore,
                    approachingScore: t.approachingScore
                ))
                continue
            }

            // --- Map Stage2 info box from detector space -> busCrop ORIGINAL space (top-left origin)
            var infoBox = Box(x1: best2.x1, y1: best2.y1, x2: best2.x2, y2: best2.y2)
            infoBox = meta2.dstToSrc(infoBox)
            infoBox = meta2.clampToSrc(infoBox)
            infoBox = expandBox(infoBox, srcW: busCropW, srcH: busCropH, margin: config.stage2CropMargin)

            if infoBox.w < 2 || infoBox.h < 2 {
                results.append(BusResult(
                    id: t.name,
                    ocrText: "",
                    bboxDetectorSpace: t.lastBoxDetector.rectXYWH(),
                    bboxOriginalTopLeft: CGRect(x: busBoxOrig.x1, y: busBoxOrig.y1, width: busBoxOrig.w, height: busBoxOrig.h),
                    confidence: t.lastScore,
                    approachingScore: t.approachingScore
                ))
                continue
            }

            // --- Crop the info region from the HIGH-RES bus crop, normalize, then OCR
            let infoCropOriginalCI = cropFromTopLeftAndNormalize(
                busCropOriginalCI,
                srcW: busCropW, srcH: busCropH,
                boxTopLeft: infoBox
            )

            var ocrText = ""
            if infoCropOriginalCI.extent.width >= 2, infoCropOriginalCI.extent.height >= 2,
               let infoCG = Self.ciContext.createCGImage(infoCropOriginalCI, from: infoCropOriginalCI.extent.integral) {
                ocrText = try await Self.ocrCGImage(
                    infoCG,
                    preset: ocrPreset,
                    recognitionLanguages: recognitionLanguages,
                    usesLanguageCorrection: usesLanguageCorrection,
                    recognitionLevel: recognitionLevel
                )
            }

            results.append(BusResult(
                id: t.name,
                ocrText: ocrText,
                bboxDetectorSpace: t.lastBoxDetector.rectXYWH(),
                bboxOriginalTopLeft: CGRect(x: busBoxOrig.x1, y: busBoxOrig.y1, width: busBoxOrig.w, height: busBoxOrig.h),
                confidence: t.lastScore,
                approachingScore: t.approachingScore
            ))
        }

        results.sort { $0.id < $1.id }
        return results
    }

    // MARK: - Stage2 picker

    private func pickBestStage2(_ det2: [Detection]) -> Detection? {
        guard !det2.isEmpty else { return nil }
        if config.preferLargestStage2Box {
            return det2.max(by: { $0.area < $1.area })
        } else {
            return det2.max(by: { $0.score < $1.score })
        }
    }

    // MARK: - Tracking

    private func matchAndUpdateTracks(detections: [Detection]) -> [Int] {
        let detBoxes: [Box] = detections.map { Box(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2) }

        // Precompute pairs by IoU threshold
        var pairs: [(iou: Double, tid: Int, di: Int)] = []
        pairs.reserveCapacity(tracks.count * max(1, detBoxes.count))

        for (tid, tr) in tracks {
            for (di, db) in detBoxes.enumerated() {
                let v = Self.iou(tr.lastBoxDetector, db)
                if v >= config.iouMatchThreshold {
                    pairs.append((v, tid, di))
                }
            }
        }

        pairs.sort { $0.iou > $1.iou }

        var usedTracks = Set<Int>()
        var detAssigned = Array(repeating: false, count: detBoxes.count)
        var matchedTrackIds: [Int] = []

        for p in pairs {
            if usedTracks.contains(p.tid) { continue }
            if detAssigned[p.di] { continue }
            usedTracks.insert(p.tid)
            detAssigned[p.di] = true
            matchedTrackIds.append(p.tid)
            updateTrack(id: p.tid, newBoxDetector: detBoxes[p.di], score: detections[p.di].score)
        }

        // New tracks for unassigned detections
        for (di, assigned) in detAssigned.enumerated() where !assigned {
            let id = nextId
            nextId += 1
            let name = "Bus\(id)"

            let box = detBoxes[di]
            let score = detections[di].score

            var tr = Track(
                numericId: id,
                name: name,
                lastBoxDetector: box,
                lastScore: score,
                lastSeenFrame: frameIndex,
                ageFrames: 1,
                areaHistory: [box.area],
                approachingScore: 0,
                isApproaching: false
            )
            computeApproach(&tr)
            tracks[id] = tr
            matchedTrackIds.append(id)
        }

        return matchedTrackIds
    }

    private func updateTrack(id: Int, newBoxDetector: Box, score: Double) {
        guard var tr = tracks[id] else { return }
        tr.lastBoxDetector = newBoxDetector
        tr.lastScore = score
        tr.lastSeenFrame = frameIndex
        tr.ageFrames += 1

        tr.areaHistory.append(newBoxDetector.area)
        if tr.areaHistory.count > config.approachWindow {
            tr.areaHistory.removeFirst(tr.areaHistory.count - config.approachWindow)
        }

        computeApproach(&tr)
        tracks[id] = tr
    }

    private func computeApproach(_ tr: inout Track) {
        let h = tr.areaHistory
        guard h.count >= config.approachMinFrames else {
            tr.approachingScore = 0
            tr.isApproaching = false
            return
        }

        let first = max(h.first ?? 0, 1e-6)
        let last = max(h.last ?? 0, 0)

        let ratio = (last / first) - 1.0
        tr.approachingScore = ratio

        // Stability: mostly increasing
        var increases = 0
        for i in 1..<h.count {
            if h[i] >= h[i - 1] { increases += 1 }
        }
        let mostlyIncreasing = increases >= max(1, (h.count - 1) * 2 / 3)

        tr.isApproaching = (ratio >= config.approachRatioThreshold) && mostlyIncreasing
    }

    private func pruneStaleTracks() {
        let cutoff = frameIndex - config.maxMissedFrames
        tracks = tracks.filter { $0.value.lastSeenFrame >= cutoff }
    }

    // MARK: - Letterbox with Meta

    /// Returns (letterboxed image in dst size, metadata for mapping boxes between spaces).
    /// Assumes src coords are pixel coords (top-left for meta usage), but CI is bottom-left;
    /// padding is symmetric so meta still maps correctly.
    private static func letterboxWithMeta(
        _ src: CIImage,
        srcW: Double,
        srcH: Double,
        dstW: Double,
        dstH: Double
    ) -> (CIImage, LetterboxMeta) {
        let scale = min(dstW / max(srcW, 1e-9), dstH / max(srcH, 1e-9))
        let resizedW = srcW * scale
        let resizedH = srcH * scale
        let padX = (dstW - resizedW) / 2.0
        let padY = (dstH - resizedH) / 2.0

        let meta = LetterboxMeta(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH, scale: scale, padX: padX, padY: padY)

        // Build the CI letterbox exactly matching meta
        let dstRect = CGRect(x: 0, y: 0, width: dstW, height: dstH)

        let scaled = src.transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
        let dx = CGFloat(padX) - scaled.extent.origin.x
        let dy = CGFloat(padY) - scaled.extent.origin.y
        let translated = scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))

        let bg = CIImage(color: .black).cropped(to: dstRect)
        let out = translated.composited(over: bg).cropped(to: dstRect)

        return (out, meta)
    }

    // MARK: - Cropping on ORIGINAL pixels (top-left boxes)

    /// Expand a box by margin fraction and clamp to src bounds.
    private func expandBox(_ b: Box, srcW: Double, srcH: Double, margin: Double) -> Box {
        guard margin > 0 else { return clampBox(b, srcW: srcW, srcH: srcH) }

        let w = b.w
        let h = b.h
        let mx = w * margin
        let my = h * margin

        let x1 = b.x1 - mx
        let y1 = b.y1 - my
        let x2 = b.x2 + mx
        let y2 = b.y2 + my

        return clampBox(Box(x1: x1, y1: y1, x2: x2, y2: y2), srcW: srcW, srcH: srcH)
    }

    private func clampBox(_ b: Box, srcW: Double, srcH: Double) -> Box {
        let x1 = max(0, min(srcW, b.x1))
        let x2 = max(0, min(srcW, b.x2))
        let y1 = max(0, min(srcH, b.y1))
        let y2 = max(0, min(srcH, b.y2))
        return Box(x1: min(x1,x2), y1: min(y1,y2), x2: max(x1,x2), y2: max(y1,y2))
    }

    /// Crop a CIImage using a box expressed in TOP-LEFT pixel coords and normalize origin to (0,0).
    private func cropFromTopLeftAndNormalize(_ src: CIImage, srcW: Double, srcH: Double, boxTopLeft: Box) -> CIImage {
        // Convert top-left coords -> CI bottom-left crop rect
        let x = boxTopLeft.x1
        let yTop = boxTopLeft.y1
        let w = boxTopLeft.w
        let h = boxTopLeft.h

        // CI y = srcH - (yTop + h)  => srcH - y2
        let ciY = srcH - (yTop + h)
        let cropRect = CGRect(x: x, y: ciY, width: w, height: h).integral

        let bounds = CGRect(x: 0, y: 0, width: srcW, height: srcH)
        let rr = cropRect.intersection(bounds).integral
        if rr.isNull || rr.width < 1 || rr.height < 1 {
            return CIImage.empty()
        }

        // Crop and normalize origin to (0,0) so downstream meta math is easy
        let cropped = src.cropped(to: rr)
        return cropped.transformed(by: CGAffineTransform(translationX: -rr.origin.x, y: -rr.origin.y))
    }

    // MARK: - PixelBuffer / Rendering

    private static func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let out = pb else {
            throw NSError(domain: "BusApproachTracker", code: 10, userInfo: [NSLocalizedDescriptionKey: "Cannot create CVPixelBuffer"])
        }
        return out
    }

    private static func render(_ image: CIImage, to pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let rect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        ciContext.render(image, to: pb, bounds: rect, colorSpace: CGColorSpaceCreateDeviceRGB())
    }

    // MARK: - YOLO parsing

    private static func readMultiArrayValue(_ arr: MLMultiArray, linearIndex: Int) -> Double {
        switch arr.dataType {
        case .double:
            return arr.dataPointer.assumingMemoryBound(to: Double.self)[linearIndex]
        case .float32:
            return Double(arr.dataPointer.assumingMemoryBound(to: Float.self)[linearIndex])
        case .float16:
            let u = arr.dataPointer.assumingMemoryBound(to: UInt16.self)[linearIndex]
            let sign = (u & 0x8000) != 0
            let exp = Int((u & 0x7C00) >> 10)
            let frac = Int(u & 0x03FF)
            var f: Float
            if exp == 0 {
                f = Float(frac) / Float(1 << 10) * powf(2, -14)
            } else if exp == 0x1F {
                f = frac == 0 ? .infinity : .nan
            } else {
                f = (1.0 + Float(frac) / Float(1 << 10)) * powf(2, Float(exp - 15))
            }
            return Double(sign ? -f : f)
        default:
            return Double(truncating: arr[linearIndex] as NSNumber)
        }
    }

    private static func parseDetections(_ arr: MLMultiArray, inputW: Double, inputH: Double) -> [Detection] {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }

        func offset(_ idx: [Int]) -> Int {
            var o = 0
            for (i, s) in zip(idx, strides) { o += i * s }
            return o
        }

        let rank = shape.count
        var N = 0
        var get: (_ i: Int, _ j: Int) -> Double = { _,_ in 0 }

        if rank == 3, shape[2] == 6 {
            N = shape[1]
            get = { i, j in readMultiArrayValue(arr, linearIndex: offset([0, i, j])) }
        } else if rank == 2, shape[1] == 6 {
            N = shape[0]
            get = { i, j in readMultiArrayValue(arr, linearIndex: offset([i, j])) }
        } else {
            let total = shape.reduce(1, *)
            N = total / 6
            get = { i, j in readMultiArrayValue(arr, linearIndex: i * 6 + j) }
        }

        var out: [Detection] = []
        out.reserveCapacity(N)

        for i in 0..<N {
            var a = get(i,0), b = get(i,1), c = get(i,2), d = get(i,3)
            let score = get(i,4)
            let cls = Int(get(i,5).rounded())

            let maxCoord = max(max(abs(a), abs(b)), max(abs(c), abs(d)))
            let isNormalized = maxCoord <= 2.0
            if isNormalized {
                a *= inputW; c *= inputW
                b *= inputH; d *= inputH
            }

            // If not xyxy, treat as cxcywh
            if c <= a || d <= b {
                let cx = a, cy = b, w = c, h = d
                a = cx - w/2.0
                b = cy - h/2.0
                c = cx + w/2.0
                d = cy + h/2.0
            }

            let x1 = min(a,c), x2 = max(a,c)
            let y1 = min(b,d), y2 = max(b,d)

            out.append(.init(x1: x1, y1: y1, x2: x2, y2: y2, score: score, cls: cls))
        }

        return out
    }

    // MARK: - IoU

    private static func iou(_ a: Box, _ b: Box) -> Double {
        let xA = max(a.x1, b.x1)
        let yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2)
        let yB = min(a.y2, b.y2)

        let interW = max(0, xB - xA)
        let interH = max(0, yB - yA)
        let inter = interW * interH

        let union = a.area + b.area - inter
        if union <= 0 { return 0 }
        return inter / union
    }

    // MARK: - OCR (improved pipeline)

    private static let ocrCIContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    private static func ocrCGImage(
        _ cgInput: CGImage,
        preset: OCRPreset,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) async throws -> String {

        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    var img = CIImage(cgImage: cgInput)

                    // 1) grayscale
                    img = img.applyingFilter("CIColorControls", parameters: [
                        kCIInputSaturationKey: 0.0,
                        kCIInputContrastKey: 1.0,
                        kCIInputBrightnessKey: 0.0
                    ])

                    // 2) flatten illumination (optional)
                    if preset.flatten {
                        let extent = img.extent
                        let minDim = min(extent.width, extent.height)
                        let radius = Float(max(10.0, Double(minDim * 0.03)))

                        let clamped = img.clampedToExtent()
                        let blur = CIFilter.gaussianBlur()
                        blur.inputImage = clamped
                        blur.radius = radius
                        let blurred = (blur.outputImage ?? clamped).cropped(to: extent)

                        img = img.applyingFilter("CIDivideBlendMode", parameters: [
                            kCIInputBackgroundImageKey: blurred
                        ])

                        img = img.applyingFilter("CIColorControls", parameters: [
                            kCIInputSaturationKey: 0.0,
                            kCIInputContrastKey: 1.4,
                            kCIInputBrightnessKey: 0.0
                        ]).cropped(to: img.extent.integral)
                    }

                    // 3) upscale
                    if preset.scale != 1 {
                        let lz = CIFilter.lanczosScaleTransform()
                        lz.inputImage = img
                        lz.scale = preset.scale
                        lz.aspectRatio = 1.0
                        let out = lz.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // 4) core tighten (pre-threshold)
                    if preset.core >= 1 {
                        let core = CIFilter.morphologyRectangleMaximum()
                        core.inputImage = img
                        core.width = preset.core
                        core.height = preset.core
                        let out = core.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // If not binarizing, feed grayscale to Vision
                    if !preset.binarize {
                        let cgOut = try renderOCRToCGImage(img)
                        let text = try visionOCR(
                            cgImage: cgOut,
                            recognitionLanguages: recognitionLanguages,
                            usesLanguageCorrection: usesLanguageCorrection,
                            recognitionLevel: recognitionLevel
                        )
                        continuation.resume(returning: text)
                        return
                    }

                    // 5) threshold
                    if let t = preset.thresh {
                        let tt = max(0, min(1, t))
                        img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": tt])
                            .cropped(to: img.extent.integral)
                    } else {
                        if #available(iOS 14.0, macOS 12.0, *) {
                            let otsu = CIFilter.colorThresholdOtsu()
                            otsu.inputImage = img
                            let out = otsu.outputImage ?? img
                            img = out.cropped(to: out.extent.integral)
                        } else {
                            img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": 0.5])
                                .cropped(to: img.extent.integral)
                        }
                    }

                    // Normalize to black text on white
                    if meanBrightness(img) < 0.5 {
                        img = img.applyingFilter("CIColorInvert").cropped(to: img.extent.integral)
                    }

                    // 6) shrink
                    if preset.shrink >= 1 {
                        let shrink = CIFilter.morphologyRectangleMaximum() // white expands => black contracts
                        shrink.inputImage = img
                        shrink.width = preset.shrink
                        shrink.height = preset.shrink
                        let out = shrink.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // 7) thinning
                    switch preset.thin {
                    case .disabled:
                        break
                    case .enabled(let maxIter):
                        img = try thinBinaryCIImage(img, maxIterations: max(0, maxIter))
                    }

                    // 8) regrow
                    if preset.regrow >= 1 {
                        let grow = CIFilter.morphologyRectangleMinimum() // black expands
                        grow.inputImage = img
                        grow.width = preset.regrow
                        grow.height = preset.regrow
                        let out = grow.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    let cgOut = try renderOCRToCGImage(img)
                    let text = try visionOCR(
                        cgImage: cgOut,
                        recognitionLanguages: recognitionLanguages,
                        usesLanguageCorrection: usesLanguageCorrection,
                        recognitionLevel: recognitionLevel
                    )
                    continuation.resume(returning: text)

                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func renderOCRToCGImage(_ img: CIImage) throws -> CGImage {
        guard let cg = ocrCIContext.createCGImage(img, from: img.extent.integral) else {
            throw NSError(domain: "OCRPipeline", code: 100, userInfo: [NSLocalizedDescriptionKey: "Cannot render OCR CIImage"])
        }
        return cg
    }

    private static func visionOCR(
        cgImage: CGImage,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = recognitionLevel
        request.usesLanguageCorrection = usesLanguageCorrection
        if let langs = recognitionLanguages, !langs.isEmpty {
            request.recognitionLanguages = langs
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let obs = request.results ?? []
        let sorted = obs.sorted {
            if $0.boundingBox.minY != $1.boundingBox.minY {
                return $0.boundingBox.minY > $1.boundingBox.minY
            }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }

        return sorted.compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
    }

    // brightness helper (no overlapping access)
    private static func meanBrightness(_ image: CIImage) -> Double {
        let avg = CIFilter.areaAverage()
        avg.inputImage = image
        avg.extent = image.extent

        guard let out = avg.outputImage,
              let cg = ocrCIContext.createCGImage(out, from: CGRect(x: 0, y: 0, width: 1, height: 1)) else { return 0.0 }

        var px = [UInt8](repeating: 0, count: 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        return px.withUnsafeMutableBytes { buf -> Double in
            guard let ctx = CGContext(
                data: buf.baseAddress,
                width: 1, height: 1,
                bitsPerComponent: 8,
                bytesPerRow: 4,
                space: cs,
                bitmapInfo: bitmapInfo
            ) else { return 0.0 }

            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: 1, height: 1))
            let p = buf.bindMemory(to: UInt8.self)
            let b = Double(p[0]), g = Double(p[1]), r = Double(p[2])
            return (r + g + b) / (3.0 * 255.0)
        }
    }

    // ---- thinning (Zhang–Suen) on rendered binary
    private static func thinBinaryCIImage(_ img: CIImage, maxIterations: Int) throws -> CIImage {
        let cg = try renderOCRToCGImage(img)
        guard let g = cgImageToGrayBytes(cg) else { return img }

        // foreground = black (byte < 128)
        var bin = [UInt8](repeating: 0, count: g.bytes.count)
        for i in 0..<g.bytes.count { bin[i] = (g.bytes[i] < 128) ? 1 : 0 }

        let thinned = zhangSuenThin(bin, width: g.width, height: g.height, maxIterations: maxIterations)

        // back to grayscale: foreground black (0), background white (255)
        var out = [UInt8](repeating: 255, count: thinned.count)
        for i in 0..<thinned.count { out[i] = (thinned[i] == 1) ? 0 : 255 }

        if let cg2 = grayBytesToCGImage(out, width: g.width, height: g.height) {
            return CIImage(cgImage: cg2).cropped(to: CGRect(x: 0, y: 0, width: g.width, height: g.height))
        }
        return img
    }

    private static func zhangSuenThin(_ src: [UInt8], width: Int, height: Int, maxIterations: Int) -> [UInt8] {
        var img = src
        if width < 3 || height < 3 { return img }

        func idx(_ x: Int, _ y: Int) -> Int { y * width + x }
        var iter = 0

        while true {
            var changed = false
            var toRemove = [Bool](repeating: false, count: img.count)

            // Step 1
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let p = idx(x, y)
                    if img[p] == 0 { continue }

                    let p2 = img[idx(x, y-1)]
                    let p3 = img[idx(x+1, y-1)]
                    let p4 = img[idx(x+1, y)]
                    let p5 = img[idx(x+1, y+1)]
                    let p6 = img[idx(x, y+1)]
                    let p7 = img[idx(x-1, y+1)]
                    let p8 = img[idx(x-1, y)]
                    let p9 = img[idx(x-1, y-1)]

                    let n = Int(p2+p3+p4+p5+p6+p7+p8+p9)
                    if n < 2 || n > 6 { continue }

                    let s =
                        ((p2 == 0 && p3 == 1) ? 1 : 0) +
                        ((p3 == 0 && p4 == 1) ? 1 : 0) +
                        ((p4 == 0 && p5 == 1) ? 1 : 0) +
                        ((p5 == 0 && p6 == 1) ? 1 : 0) +
                        ((p6 == 0 && p7 == 1) ? 1 : 0) +
                        ((p7 == 0 && p8 == 1) ? 1 : 0) +
                        ((p8 == 0 && p9 == 1) ? 1 : 0) +
                        ((p9 == 0 && p2 == 1) ? 1 : 0)
                    if s != 1 { continue }

                    if (p2 * p4 * p6) != 0 { continue }
                    if (p4 * p6 * p8) != 0 { continue }

                    toRemove[p] = true
                }
            }
            for i in 0..<toRemove.count where toRemove[i] {
                img[i] = 0
                changed = true
            }

            toRemove = [Bool](repeating: false, count: img.count)

            // Step 2
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let p = idx(x, y)
                    if img[p] == 0 { continue }

                    let p2 = img[idx(x, y-1)]
                    let p3 = img[idx(x+1, y-1)]
                    let p4 = img[idx(x+1, y)]
                    let p5 = img[idx(x+1, y+1)]
                    let p6 = img[idx(x, y+1)]
                    let p7 = img[idx(x-1, y+1)]
                    let p8 = img[idx(x-1, y)]
                    let p9 = img[idx(x-1, y-1)]

                    let n = Int(p2+p3+p4+p5+p6+p7+p8+p9)
                    if n < 2 || n > 6 { continue }

                    let s =
                        ((p2 == 0 && p3 == 1) ? 1 : 0) +
                        ((p3 == 0 && p4 == 1) ? 1 : 0) +
                        ((p4 == 0 && p5 == 1) ? 1 : 0) +
                        ((p5 == 0 && p6 == 1) ? 1 : 0) +
                        ((p6 == 0 && p7 == 1) ? 1 : 0) +
                        ((p7 == 0 && p8 == 1) ? 1 : 0) +
                        ((p8 == 0 && p9 == 1) ? 1 : 0) +
                        ((p9 == 0 && p2 == 1) ? 1 : 0)
                    if s != 1 { continue }

                    if (p2 * p4 * p8) != 0 { continue }
                    if (p2 * p6 * p8) != 0 { continue }

                    toRemove[p] = true
                }
            }
            for i in 0..<toRemove.count where toRemove[i] {
                img[i] = 0
                changed = true
            }

            iter += 1
            if !changed { break }
            if maxIterations > 0 && iter >= maxIterations { break }
        }

        return img
    }

    private static func cgImageToGrayBytes(_ cg: CGImage) -> (bytes: [UInt8], width: Int, height: Int)? {
        let w = cg.width
        let h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h)

        let ok = buf.withUnsafeMutableBytes { ptr -> Bool in
            guard let ctx = CGContext(
                data: ptr.baseAddress,
                width: w, height: h,
                bitsPerComponent: 8,
                bytesPerRow: w,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? (buf, w, h) : nil
    }

    private static func grayBytesToCGImage(_ bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        var data = bytes
        guard let provider = CGDataProvider(data: Data(bytes: &data, count: data.count) as CFData) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}

// MARK: - Optional adapters (UIImage / NSImage -> CGImage)

#if canImport(UIKit)
import UIKit

public extension UIImage {
    func toCGImage() -> CGImage? {
        if let cg = self.cgImage { return cg }
        if let ci = self.ciImage {
            return CIContext().createCGImage(ci, from: ci.extent.integral)
        }
        let w = Int(size.width), h = Int(size.height)
        guard w > 0, h > 0 else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs, bitmapInfo: bitmapInfo) else { return nil }
        UIGraphicsPushContext(ctx); defer { UIGraphicsPopContext() }
        self.draw(in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }
}
#endif

#if canImport(AppKit)
import AppKit

public extension NSImage {
    func toCGImage() -> CGImage? {
        var rect = CGRect(origin: .zero, size: self.size)
        return self.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
#endif
