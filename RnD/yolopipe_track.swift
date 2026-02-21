#!/usr/bin/env swift
//
//  yolopipe_track.swift
//
//  Terminal test app (single Swift file) that matches the Xcode “BusApproachTracker_HiResCrops” behavior:
//   - Stage1 (bus detect) on fixed detector size (default 512x896) using letterbox(original -> detector)
//   - Multi-bus tracking via IoU in detector space
//   - “Approaching” via bbox area growth over a window
//   - For approaching buses only: Stage2 (info detect) on letterbox(busCrop -> detector)
//   - BUT: all crops (bus crop + info crop) are taken from ORIGINAL high-res pixels by un-letterboxing boxes back
//   - OCR runs on the highest-res possible info crop (from original pixels), then preprocessing + Vision OCR
//
//  Inputs are name=value (space separated), e.g.:
//    swift yolopipe_track.swift \
//      video=/path/in.mov \
//      dest=/path/outdir \
//      yolo=/path/yolo26sINT8.mlpackage \
//      yoloCustom=/path/busInfoYolo26sINT8.mlpackage \
//      targetW=512 targetH=896 \
//      busClass=5 s1conf=0.51 s2conf=0.51 \
//      stage1CropMargin=0.00 stage2CropMargin=0.00 \
//      iou=0.35 maxMissed=8 window=5 minFrames=4 approach=0.10 \
//      ocrScale=1 ocrFlatten=true ocrCore=1 ocrBinarize=false \
//      langs=system
//
//  Output:
//   - dest/frames/frame_000001/ padded.png + <BusN>_bus.png + <BusN>_info.png + frame.json
//   - dest/summary.json
//

import Foundation
import AVFoundation
import CoreML
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Helpers (Args)

func parseArgs(_ argv: [String]) -> [String: String] {
    var dict: [String: String] = [:]
    for a in argv {
        guard let eq = a.firstIndex(of: "=") else { continue }   // (intentionally not strict)
        let k = String(a[..<eq]).trimmingCharacters(in: .whitespacesAndNewlines)
        let v = String(a[a.index(after: eq)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        if !k.isEmpty { dict[k] = v }
    }
    return dict
}

func boolValue(_ s: String?, default d: Bool) -> Bool {
    guard let s = s?.lowercased() else { return d }
    if ["1","true","yes","y","on"].contains(s) { return true }
    if ["0","false","no","n","off"].contains(s) { return false }
    return d
}

func intValue(_ s: String?, default d: Int) -> Int {
    guard let s = s, let v = Int(s) else { return d }
    return v
}

func doubleValue(_ s: String?, default d: Double) -> Double {
    guard let s = s, let v = Double(s) else { return d }
    return v
}

func floatValue(_ s: String?, default d: Float) -> Float {
    guard let s = s, let v = Float(s) else { return d }
    return v
}

func usageAndExit(_ msg: String? = nil) -> Never {
    if let msg { fputs("Error: \(msg)\n\n", stderr) }
    print("""
    Usage:
      swift yolopipe_track.swift video=... dest=... yolo=... yoloCustom=... [options...]

    Required:
      video=PATH            Input video file
      dest=DIR              Destination output folder
      yolo=PATH             Stage1 model path (.mlpackage/.mlmodel/.mlmodelc)
      yoloCustom=PATH       Stage2 model path (.mlpackage/.mlmodel/.mlmodelc)

    Detector:
      targetW=512 targetH=896
      busClass=5
      s1conf=0.51
      s2conf=0.51
      preferLargestStage2=false

    Hi-res crop margins (fraction of box size):
      stage1CropMargin=0.00
      stage2CropMargin=0.00

    Tracking options:
      iou=0.35
      maxMissed=8
      window=5
      minFrames=4
      approach=0.10

    OCR options (preprocess + Vision):
      ocrScale=1
      ocrFlatten=true
      ocrCore=1
      ocrBinarize=false
      ocrThresh=0.70          (only used if ocrBinarize=true; optional)
      ocrShrink=0
      ocrThin=disabled        (or: enabled:0 for until-stable, enabled:12 for max 12)
      ocrRegrow=0
      langs=system            (or: langs=pt-PT,en-US)
      usesLanguageCorrection=false
      recognitionLevel=accurate|fast

    Output controls:
      saveImages=true         (saves stage1Input padded + hi-res bus/info crops for approaching buses)
      printPerFrame=false     (prints per-frame approach + OCR)
    """)
    exit(2)
}

// MARK: - OCR Preset

struct OCRPreset {
    enum Thin {
        case disabled
        case enabled(maxIterations: Int) // 0 = until stable
    }

    var scale: Float
    var flatten: Bool
    var core: Float
    var binarize: Bool
    var thresh: Float?
    var shrink: Float
    var thin: Thin
    var regrow: Float

    static let disabled = OCRPreset(
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

// MARK: - CI / Image IO helpers

final class SharedCI {
    static let context: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()
}

func ensureDir(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
}

func savePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "savePNG", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create CGImageDestination"])
    }
    CGImageDestinationAddImage(dest, image, nil)
    if !CGImageDestinationFinalize(dest) {
        throw NSError(domain: "savePNG", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot finalize image destination"])
    }
}

func saveJSON(_ obj: Any, to url: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url, options: .atomic)
}

func renderToCGImage(_ img: CIImage) throws -> CGImage {
    let rect = img.extent.integral
    guard let cg = SharedCI.context.createCGImage(img, from: rect) else {
        throw NSError(domain: "renderToCGImage", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot render CIImage to CGImage"])
    }
    return cg
}

// MARK: - PixelBuffer / Rendering

func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)
    ]
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    guard status == kCVReturnSuccess, let out = pb else {
        throw NSError(domain: "makePixelBuffer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create CVPixelBuffer"])
    }
    return out
}

func renderToPixelBuffer(_ img: CIImage, pb: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let rect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
    SharedCI.context.render(img, to: pb, bounds: rect, colorSpace: CGColorSpaceCreateDeviceRGB())
}

// MARK: - Box + IoU

struct Box {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double

    var w: Double { max(0, x2 - x1) }
    var h: Double { max(0, y2 - y1) }
    var area: Double { w * h }
}

func iou(_ a: Box, _ b: Box) -> Double {
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

// MARK: - Letterbox with Meta (detector <-> source mapping, TOP-LEFT coords)

struct LetterboxMeta {
    let srcW: Double
    let srcH: Double
    let dstW: Double
    let dstH: Double
    let scale: Double
    let padX: Double
    let padY: Double

    // dst(detector) -> src(original/crop), both in TOP-LEFT coords
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

func letterboxWithMeta(_ image: CIImage, srcW: Double, srcH: Double, targetW: Int, targetH: Int) -> (CIImage, LetterboxMeta) {
    let dstW = Double(targetW), dstH = Double(targetH)
    let scale = min(dstW / max(srcW, 1e-9), dstH / max(srcH, 1e-9))
    let resizedW = srcW * scale
    let resizedH = srcH * scale
    let padX = (dstW - resizedW) / 2.0
    let padY = (dstH - resizedH) / 2.0

    let meta = LetterboxMeta(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH, scale: scale, padX: padX, padY: padY)

    let dstRect = CGRect(x: 0, y: 0, width: dstW, height: dstH)

    let scaled = image.transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
    let dx = CGFloat(padX) - scaled.extent.origin.x
    let dy = CGFloat(padY) - scaled.extent.origin.y
    let translated = scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))

    let bg = CIImage(color: .black).cropped(to: dstRect)
    let out = translated.composited(over: bg).cropped(to: dstRect)
    return (out, meta)
}

// MARK: - Hi-res crop helpers (TOP-LEFT boxes on ORIGINAL pixels)

func clampBox(_ b: Box, srcW: Double, srcH: Double) -> Box {
    let x1 = max(0, min(srcW, b.x1))
    let x2 = max(0, min(srcW, b.x2))
    let y1 = max(0, min(srcH, b.y1))
    let y2 = max(0, min(srcH, b.y2))
    return Box(x1: min(x1,x2), y1: min(y1,y2), x2: max(x1,x2), y2: max(y1,y2))
}

func expandBox(_ b: Box, srcW: Double, srcH: Double, margin: Double) -> Box {
    guard margin > 0 else { return clampBox(b, srcW: srcW, srcH: srcH) }
    let mx = b.w * margin
    let my = b.h * margin
    return clampBox(Box(x1: b.x1 - mx, y1: b.y1 - my, x2: b.x2 + mx, y2: b.y2 + my),
                    srcW: srcW, srcH: srcH)
}

// Crop CIImage using TOP-LEFT coords; normalize origin to (0,0)
func cropFromTopLeftAndNormalize(_ src: CIImage, srcW: Double, srcH: Double, boxTopLeft: Box) -> CIImage {
    let x = boxTopLeft.x1
    let yTop = boxTopLeft.y1
    let w = boxTopLeft.w
    let h = boxTopLeft.h

    // CI is bottom-left origin: yCI = srcH - (yTop + h)
    let ciY = srcH - (yTop + h)
    let cropRect = CGRect(x: x, y: ciY, width: w, height: h).integral

    let bounds = CGRect(x: 0, y: 0, width: srcW, height: srcH)
    let rr = cropRect.intersection(bounds).integral
    if rr.isNull || rr.width < 1 || rr.height < 1 { return CIImage.empty() }

    let cropped = src.cropped(to: rr)
    return cropped.transformed(by: CGAffineTransform(translationX: -rr.origin.x, y: -rr.origin.y))
}

// MARK: - CoreML model loading (mlpackage/mlmodel/mlmodelc)

func compileIfNeeded(_ url: URL) throws -> URL {
    let ext = url.pathExtension.lowercased()
    if ext == "mlmodelc" { return url }
    return try MLModel.compileModel(at: url) // .mlmodel or .mlpackage
}

final class YOLOModel {
    enum BoxOrigin { case topLeft, bottomLeft }

    let model: MLModel
    let inputName: String
    let outputName: String
    let boxOrigin: BoxOrigin

    init(modelURL: URL, boxOrigin: BoxOrigin = .topLeft, computeUnits: MLComputeUnits = .all) throws {
        self.boxOrigin = boxOrigin
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        let compiledURL = try compileIfNeeded(modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: cfg)

        let inputs = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard !inputs.isEmpty else { throw NSError(domain: "YOLOModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model has no inputs"]) }
        self.inputName = inputs.contains("image") ? "image" : inputs[0]

        let outputs = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard !outputs.isEmpty else { throw NSError(domain: "YOLOModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model has no outputs"]) }
        self.outputName = outputs[0]
    }

    func predict(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let out = try model.prediction(from: provider)
        guard let fv = out.featureValue(for: outputName), let arr = fv.multiArrayValue else {
            throw NSError(domain: "YOLOModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Output \(outputName) is not MLMultiArray"])
        }
        return arr
    }
}

// MARK: - YOLO output parsing

struct Detection {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double
    var score: Double
    var cls: Int
    var area: Double { max(0, x2 - x1) * max(0, y2 - y1) }
}

func readMultiArrayValue(_ arr: MLMultiArray, linearIndex: Int) -> Double {
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

// IMPORTANT: returns TOP-LEFT coords (converts if model origin is bottom-left)
func parseDetections(_ arr: MLMultiArray, inputW: Double, inputH: Double, boxOrigin: YOLOModel.BoxOrigin) -> [Detection] {
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

        var x1 = min(a,c), x2 = max(a,c)
        var y1 = min(b,d), y2 = max(b,d)

        // Convert bottom-left -> top-left if needed
        if boxOrigin == .bottomLeft {
            let ny1 = inputH - y2
            let ny2 = inputH - y1
            y1 = min(ny1, ny2)
            y2 = max(ny1, ny2)
        }

        out.append(.init(x1: x1, y1: y1, x2: x2, y2: y2, score: score, cls: cls))
    }

    return out
}

// MARK: - OCR pipeline (sync, matches your Xcode logic)

final class OCRPipeline {
    static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    static func ocr(
        cgImage: CGImage,
        preset: OCRPreset,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) throws -> String {
        var img = CIImage(cgImage: cgImage)

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

        // If binarize=false, feed grayscale to Vision
        if !preset.binarize {
            let cgOut = try renderToCG(img)
            return try visionOCR(
                cgImage: cgOut,
                recognitionLanguages: recognitionLanguages,
                usesLanguageCorrection: usesLanguageCorrection,
                recognitionLevel: recognitionLevel
            )
        }

        // 5) threshold
        if let t = preset.thresh {
            let tt = max(0, min(1, t))
            img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": tt])
                .cropped(to: img.extent.integral)
        } else {
            if #available(macOS 12.0, *) {
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

        // 6) optional shrink
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

        let cgOut = try renderToCG(img)
        return try visionOCR(
            cgImage: cgOut,
            recognitionLanguages: recognitionLanguages,
            usesLanguageCorrection: usesLanguageCorrection,
            recognitionLevel: recognitionLevel
        )
    }

    private static func renderToCG(_ img: CIImage) throws -> CGImage {
        guard let cg = ciContext.createCGImage(img, from: img.extent.integral) else {
            throw NSError(domain: "OCRPipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot render OCR CIImage"])
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

    private static func meanBrightness(_ image: CIImage) -> Double {
        let avg = CIFilter.areaAverage()
        avg.inputImage = image
        avg.extent = image.extent

        guard let out = avg.outputImage,
              let cg = ciContext.createCGImage(out, from: CGRect(x: 0, y: 0, width: 1, height: 1)) else { return 0.0 }

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
        let cg = try renderToCG(img)
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

// MARK: - Tracking + Approaching

struct TrackerConfig {
    var targetW: Int = 512
    var targetH: Int = 896

    var busClass: Int = 5
    var s1conf: Double = 0.50
    var s2conf: Double = 0.50
    var preferLargestStage2: Bool = false

    var iouMatchThreshold: Double = 0.35
    var maxMissedFrames: Int = 8

    var approachWindow: Int = 5
    var approachMinFrames: Int = 4
    var approachRatioThreshold: Double = 0.10

    // Hi-res crop margins (fraction of box size)
    var stage1CropMargin: Double = 0.00
    var stage2CropMargin: Double = 0.00
}

struct TrackSnapshot {
    let id: Int
    let name: String
    let box: Box
    let score: Double
    let lastSeenFrame: Int
    let ageFrames: Int
    let areaHistory: [Double]
    let approachingScore: Double
    let isApproaching: Bool
}

final class BusApproachTrackerTerminal {
    struct BusResult {
        let name: String
        let boxDetector: Box            // detector-space box (same as tracking)
        let confidence: Double
        let approachingScore: Double
        let ocrText: String
        let busCropCG: CGImage?         // HI-RES bus crop from original
        let infoCropCG: CGImage?        // HI-RES info crop from original bus crop
    }

    private let stage1: YOLOModel
    private let stage2: YOLOModel
    private let config: TrackerConfig

    private var nextId: Int = 1
    private var frameIndex: Int = 0

    private struct Track {
        let id: Int
        let name: String
        var lastBox: Box
        var lastScore: Double
        var lastSeenFrame: Int
        var ageFrames: Int
        var areaHistory: [Double]
        var approachingScore: Double
        var isApproaching: Bool
    }

    private var tracks: [Int: Track] = [:]

    init(stage1: YOLOModel, stage2: YOLOModel, config: TrackerConfig) {
        self.stage1 = stage1
        self.stage2 = stage2
        self.config = config
    }

    func processFrame(
        frameCG: CGImage,
        ocrPreset: OCRPreset,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) throws -> (stage1Detections: [Detection], tracks: [TrackSnapshot], approaching: [BusResult], stage1InputFrame: CGImage) {

        frameIndex += 1

        // ORIGINAL hi-res frame CI
        let originalW = Double(frameCG.width)
        let originalH = Double(frameCG.height)
        let originalCI = CIImage(cgImage: frameCG)

        // Stage1 input: letterbox ORIGINAL -> detector size (with meta for reverse mapping)
        let (stage1InputCI, meta1) = letterboxWithMeta(
            originalCI,
            srcW: originalW, srcH: originalH,
            targetW: config.targetW, targetH: config.targetH
        )
        let stage1InputCG = try renderToCGImage(stage1InputCI)

        // Stage1 inference (detector-sized)
        let pb1 = try makePixelBuffer(width: config.targetW, height: config.targetH)
        renderToPixelBuffer(stage1InputCI, pb: pb1)

        let out1 = try stage1.predict(pixelBuffer: pb1)
        var det1 = parseDetections(out1,
                                   inputW: Double(config.targetW),
                                   inputH: Double(config.targetH),
                                   boxOrigin: stage1.boxOrigin)

        det1 = det1.filter { $0.cls == config.busClass && $0.score >= config.s1conf }
                   .sorted { $0.score > $1.score }

        // Tracking update in detector space
        let matchedThisFrame = matchAndUpdateTracks(detections: det1)

        // Prune stale
        pruneStaleTracks()

        // Snapshots
        let snapshots: [TrackSnapshot] = tracks.values.sorted { $0.id < $1.id }.map { t in
            TrackSnapshot(
                id: t.id,
                name: t.name,
                box: t.lastBox,
                score: t.lastScore,
                lastSeenFrame: t.lastSeenFrame,
                ageFrames: t.ageFrames,
                areaHistory: t.areaHistory,
                approachingScore: t.approachingScore,
                isApproaching: t.isApproaching
            )
        }

        // For approaching tracks seen this frame -> Stage2 + OCR using HI-RES crops
        var results: [BusResult] = []

        for tid in matchedThisFrame.sorted() {
            guard let t = tracks[tid], t.isApproaching else { continue }

            // Map Stage1 detector box -> ORIGINAL top-left coords, clamp + optional margin
            var busBoxOrig = meta1.dstToSrc(t.lastBox)
            busBoxOrig = meta1.clampToSrc(busBoxOrig)
            busBoxOrig = expandBox(busBoxOrig, srcW: originalW, srcH: originalH, margin: config.stage1CropMargin)

            if busBoxOrig.w < 2 || busBoxOrig.h < 2 { continue }

            // HI-RES bus crop from ORIGINAL (normalized origin)
            let busCropOriginalCI = cropFromTopLeftAndNormalize(
                originalCI,
                srcW: originalW, srcH: originalH,
                boxTopLeft: busBoxOrig
            )
            let busCropW = Double(busCropOriginalCI.extent.width)
            let busCropH = Double(busCropOriginalCI.extent.height)
            if busCropW < 2 || busCropH < 2 { continue }

            let busCropCG = try? renderToCGImage(busCropOriginalCI)

            // Stage2 input: letterbox BUS CROP -> detector size (with meta2)
            let (stage2InputCI, meta2) = letterboxWithMeta(
                busCropOriginalCI,
                srcW: busCropW, srcH: busCropH,
                targetW: config.targetW, targetH: config.targetH
            )

            let pb2 = try makePixelBuffer(width: config.targetW, height: config.targetH)
            renderToPixelBuffer(stage2InputCI, pb: pb2)

            let out2 = try stage2.predict(pixelBuffer: pb2)
            var det2 = parseDetections(out2,
                                       inputW: Double(config.targetW),
                                       inputH: Double(config.targetH),
                                       boxOrigin: stage2.boxOrigin)
            det2 = det2.filter { $0.score >= config.s2conf }

            let best2: Detection? = {
                if det2.isEmpty { return nil }
                if config.preferLargestStage2 { return det2.max(by: { $0.area < $1.area }) }
                return det2.max(by: { $0.score < $1.score })
            }()

            var infoCropCG: CGImage? = nil
            var ocrText = ""

            if let b2 = best2 {
                // Map Stage2 detector box -> BUS CROP original top-left coords, clamp + margin
                var infoBox = Box(x1: b2.x1, y1: b2.y1, x2: b2.x2, y2: b2.y2)
                infoBox = meta2.dstToSrc(infoBox)
                infoBox = meta2.clampToSrc(infoBox)
                infoBox = expandBox(infoBox, srcW: busCropW, srcH: busCropH, margin: config.stage2CropMargin)

                if infoBox.w >= 2, infoBox.h >= 2 {
                    // HI-RES info crop from HI-RES bus crop
                    let infoCropOriginalCI = cropFromTopLeftAndNormalize(
                        busCropOriginalCI,
                        srcW: busCropW, srcH: busCropH,
                        boxTopLeft: infoBox
                    )

                    if infoCropOriginalCI.extent.width >= 2, infoCropOriginalCI.extent.height >= 2 {
                        infoCropCG = try? renderToCGImage(infoCropOriginalCI)
                        if let cg = infoCropCG {
                            ocrText = (try? OCRPipeline.ocr(
                                cgImage: cg,
                                preset: ocrPreset,
                                recognitionLanguages: recognitionLanguages,
                                usesLanguageCorrection: usesLanguageCorrection,
                                recognitionLevel: recognitionLevel
                            )) ?? ""
                        }
                    }
                }
            }

            results.append(BusResult(
                name: t.name,
                boxDetector: t.lastBox,
                confidence: t.lastScore,
                approachingScore: t.approachingScore,
                ocrText: ocrText,
                busCropCG: busCropCG,
                infoCropCG: infoCropCG
            ))
        }

        results.sort { $0.name < $1.name }
        return (det1, snapshots, results, stage1InputCG)
    }

    // --- tracking
    private func matchAndUpdateTracks(detections: [Detection]) -> [Int] {
        let detBoxes = detections.map { Box(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2) }

        var pairs: [(iou: Double, tid: Int, di: Int)] = []
        pairs.reserveCapacity(tracks.count * max(1, detBoxes.count))

        for (tid, tr) in tracks {
            for (di, db) in detBoxes.enumerated() {
                let v = iou(tr.lastBox, db)
                if v >= config.iouMatchThreshold {
                    pairs.append((v, tid, di))
                }
            }
        }

        pairs.sort { $0.iou > $1.iou }

        var usedTracks = Set<Int>()
        var detAssigned = Array(repeating: false, count: detBoxes.count)
        var matchedThisFrame: [Int] = []

        for p in pairs {
            if usedTracks.contains(p.tid) { continue }
            if detAssigned[p.di] { continue }
            usedTracks.insert(p.tid)
            detAssigned[p.di] = true
            matchedThisFrame.append(p.tid)
            updateTrack(id: p.tid, newBox: detBoxes[p.di], score: detections[p.di].score)
        }

        for (di, assigned) in detAssigned.enumerated() where !assigned {
            let id = nextId
            nextId += 1
            let name = "Bus\(id)"
            let box = detBoxes[di]
            let score = detections[di].score
            var tr = Track(
                id: id,
                name: name,
                lastBox: box,
                lastScore: score,
                lastSeenFrame: frameIndex,
                ageFrames: 1,
                areaHistory: [box.area],
                approachingScore: 0,
                isApproaching: false
            )
            computeApproach(&tr)
            tracks[id] = tr
            matchedThisFrame.append(id)
        }

        return matchedThisFrame
    }

    private func updateTrack(id: Int, newBox: Box, score: Double) {
        guard var tr = tracks[id] else { return }
        tr.lastBox = newBox
        tr.lastScore = score
        tr.lastSeenFrame = frameIndex
        tr.ageFrames += 1
        tr.areaHistory.append(newBox.area)
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

        var increases = 0
        for i in 1..<h.count { if h[i] >= h[i-1] { increases += 1 } }
        let mostlyIncreasing = increases >= max(1, (h.count - 1) * 2 / 3)

        tr.isApproaching = (ratio >= config.approachRatioThreshold) && mostlyIncreasing
    }

    private func pruneStaleTracks() {
        let cutoff = frameIndex - config.maxMissedFrames
        tracks = tracks.filter { $0.value.lastSeenFrame >= cutoff }
    }
}

// MARK: - Main

let args = parseArgs(Array(CommandLine.arguments.dropFirst()))

guard let videoPath = args["video"], !videoPath.isEmpty else { usageAndExit("Missing video=") }
guard let destPath  = args["dest"],  !destPath.isEmpty  else { usageAndExit("Missing dest=") }
guard let yoloPath  = args["yolo"],  !yoloPath.isEmpty  else { usageAndExit("Missing yolo=") }
guard let y2Path    = args["yoloCustom"], !y2Path.isEmpty else { usageAndExit("Missing yoloCustom=") }

let targetW = intValue(args["targetW"], default: 512)
let targetH = intValue(args["targetH"], default: 896)

let busClass = intValue(args["busClass"], default: 5)
let s1conf = doubleValue(args["s1conf"], default: 0.51)
let s2conf = doubleValue(args["s2conf"], default: 0.51)
let preferLargestStage2 = boolValue(args["preferLargestStage2"], default: false)

let stage1CropMargin = doubleValue(args["stage1CropMargin"], default: 0.0)
let stage2CropMargin = doubleValue(args["stage2CropMargin"], default: 0.0)

let iouThr = doubleValue(args["iou"], default: 0.35)
let maxMissed = intValue(args["maxMissed"], default: 8)
let window = intValue(args["window"], default: 5)
let minFrames = intValue(args["minFrames"], default: 4)
let approach = doubleValue(args["approach"], default: 0.10)

let saveImages = boolValue(args["saveImages"], default: true)
let printPerFrame = boolValue(args["printPerFrame"], default: false)

// OCR params
let ocrScale = floatValue(args["ocrScale"], default: 1)
let ocrFlatten = boolValue(args["ocrFlatten"], default: true)
let ocrCore = floatValue(args["ocrCore"], default: 1)
let ocrBinarize = boolValue(args["ocrBinarize"], default: false)
let ocrThresh: Float? = {
    guard let s = args["ocrThresh"], let v = Float(s) else { return nil }
    return v
}()
let ocrShrink = floatValue(args["ocrShrink"], default: 0)
let ocrRegrow = floatValue(args["ocrRegrow"], default: 0)

let ocrThin: OCRPreset.Thin = {
    let raw = (args["ocrThin"] ?? "disabled").lowercased()
    if raw == "disabled" { return .disabled }
    if raw == "enabled" { return .enabled(maxIterations: 0) }
    if raw.hasPrefix("enabled:") {
        let tail = raw.replacingOccurrences(of: "enabled:", with: "")
        let it = Int(tail) ?? 0
        return .enabled(maxIterations: max(0, it))
    }
    return .disabled
}()

let ocrPreset = OCRPreset(
    scale: ocrScale,
    flatten: ocrFlatten,
    core: ocrCore,
    binarize: ocrBinarize,
    thresh: ocrThresh,
    shrink: ocrShrink,
    thin: ocrThin,
    regrow: ocrRegrow
)

let recognitionLanguages: [String]? = {
    let raw = (args["langs"] ?? "system")
    if raw.lowercased() == "system" { return Locale.preferredLanguages }
    let parts = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    return parts.isEmpty ? nil : parts
}()

let usesLanguageCorrection = boolValue(args["usesLanguageCorrection"], default: false)
let recognitionLevel: VNRequestTextRecognitionLevel = (args["recognitionLevel"]?.lowercased() == "fast") ? .fast : .accurate

let videoURL = URL(fileURLWithPath: videoPath)
let destURL = URL(fileURLWithPath: destPath)
let yoloURL = URL(fileURLWithPath: yoloPath)
let y2URL = URL(fileURLWithPath: y2Path)

do {
    try ensureDir(destURL)
    let framesURL = destURL.appendingPathComponent("frames", isDirectory: true)
    try ensureDir(framesURL)

    print("Loading models...")
    let tModel0 = CFAbsoluteTimeGetCurrent()
    let stage1 = try YOLOModel(modelURL: yoloURL, boxOrigin: .topLeft, computeUnits: .all)
    let stage2 = try YOLOModel(modelURL: y2URL, boxOrigin: .topLeft, computeUnits: .all)
    let tModel1 = CFAbsoluteTimeGetCurrent()
    print(String(format: "Models loaded in %.3fs", (tModel1 - tModel0)))

    var cfg = TrackerConfig()
    cfg.targetW = targetW
    cfg.targetH = targetH
    cfg.busClass = busClass
    cfg.s1conf = s1conf
    cfg.s2conf = s2conf
    cfg.preferLargestStage2 = preferLargestStage2
    cfg.iouMatchThreshold = iouThr
    cfg.maxMissedFrames = maxMissed
    cfg.approachWindow = window
    cfg.approachMinFrames = minFrames
    cfg.approachRatioThreshold = approach
    cfg.stage1CropMargin = stage1CropMargin
    cfg.stage2CropMargin = stage2CropMargin

    let tracker = BusApproachTrackerTerminal(stage1: stage1, stage2: stage2, config: cfg)

    // Video reader
    let asset = AVURLAsset(url: videoURL)
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw NSError(domain: "yolopipe_track", code: 1, userInfo: [NSLocalizedDescriptionKey: "No video track found"])
    }

    let reader = try AVAssetReader(asset: asset)
    let outputSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw NSError(domain: "yolopipe_track", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot add track output"])
    }
    reader.add(output)
    reader.startReading()

    let t0 = CFAbsoluteTimeGetCurrent()
    var frames = 0
    var procTimeSum: Double = 0

    var summaryFrames: [[String: Any]] = []

    while reader.status == .reading {
        autoreleasepool {
            guard let sb = output.copyNextSampleBuffer(),
                  let pb = CMSampleBufferGetImageBuffer(sb) else {
                return
            }

            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let seconds = CMTimeGetSeconds(pts)

            // Oriented CIImage
            var frameCI = CIImage(cvPixelBuffer: pb).transformed(by: track.preferredTransform)
            // normalize origin to (0,0)
            let e = frameCI.extent
            frameCI = frameCI.transformed(by: CGAffineTransform(translationX: -e.origin.x, y: -e.origin.y))

            // Render to CGImage for tracker input
            guard let frameCG = SharedCI.context.createCGImage(frameCI, from: frameCI.extent.integral) else {
                return
            }

            frames += 1
            let frameIndex = frames
            let frameFolder = framesURL.appendingPathComponent(String(format: "frame_%06d", frameIndex), isDirectory: true)
            do { try ensureDir(frameFolder) } catch { /* ignore */ }

            let tFrame0 = CFAbsoluteTimeGetCurrent()
            do {
                let out = try tracker.processFrame(
                    frameCG: frameCG,
                    ocrPreset: ocrPreset,
                    recognitionLanguages: recognitionLanguages,
                    usesLanguageCorrection: usesLanguageCorrection,
                    recognitionLevel: recognitionLevel
                )
                let tFrame1 = CFAbsoluteTimeGetCurrent()
                let dt = tFrame1 - tFrame0
                procTimeSum += dt

                // Save images (optional)
                var saved: [String: Any] = [:]
                if saveImages {
                    // stage1 input (detector-sized letterboxed original)
                    let paddedURL = frameFolder.appendingPathComponent("padded.png")
                    try? savePNG(out.stage1InputFrame, to: paddedURL)
                    saved["padded"] = paddedURL.lastPathComponent

                    // approaching bus crops (HI-RES)
                    var busImgs: [[String: Any]] = []
                    for b in out.approaching {
                        var entry: [String: Any] = ["id": b.name]
                        if let busCG = b.busCropCG {
                            let u = frameFolder.appendingPathComponent("\(b.name)_bus.png")
                            try? savePNG(busCG, to: u)
                            entry["busCrop"] = u.lastPathComponent
                        }
                        if let infoCG = b.infoCropCG {
                            let u = frameFolder.appendingPathComponent("\(b.name)_info.png")
                            try? savePNG(infoCG, to: u)
                            entry["infoCrop"] = u.lastPathComponent
                        }
                        busImgs.append(entry)
                    }
                    saved["crops"] = busImgs
                }

                // Per-frame JSON
                let detJSON: [[String: Any]] = out.stage1Detections.map {
                    ["x1": $0.x1, "y1": $0.y1, "x2": $0.x2, "y2": $0.y2, "score": $0.score, "cls": $0.cls]
                }
                let tracksJSON: [[String: Any]] = out.tracks.map {
                    [
                        "id": $0.id,
                        "name": $0.name,
                        "bboxDetector": ["x1": $0.box.x1, "y1": $0.box.y1, "x2": $0.box.x2, "y2": $0.box.y2],
                        "score": $0.score,
                        "ageFrames": $0.ageFrames,
                        "lastSeenFrame": $0.lastSeenFrame,
                        "areaHistory": $0.areaHistory,
                        "approachingScore": $0.approachingScore,
                        "isApproaching": $0.isApproaching
                    ]
                }
                let approachingJSON: [[String: Any]] = out.approaching.map {
                    [
                        "id": $0.name,
                        "bboxDetector": ["x1": $0.boxDetector.x1, "y1": $0.boxDetector.y1, "x2": $0.boxDetector.x2, "y2": $0.boxDetector.y2],
                        "confidence": $0.confidence,
                        "approachingScore": $0.approachingScore,
                        "ocrText": $0.ocrText
                    ]
                }

                let frameJSON: [String: Any] = [
                    "frameIndex": frameIndex,
                    "timestampSeconds": seconds,
                    "processingSeconds": dt,
                    "stage1Detections": detJSON,
                    "tracks": tracksJSON,
                    "approachingBuses": approachingJSON,
                    "saved": saved
                ]

                let jsonURL = frameFolder.appendingPathComponent("frame.json")
                try? saveJSON(frameJSON, to: jsonURL)

                summaryFrames.append([
                    "frameIndex": frameIndex,
                    "timestampSeconds": seconds,
                    "processingSeconds": dt,
                    "approachingBuses": approachingJSON
                ])

                if printPerFrame {
                    if approachingJSON.isEmpty {
                        print(String(format: "frame %06d  %.1f ms  approaching: none", frameIndex, dt * 1000.0))
                    } else {
                        for b in out.approaching {
                            let oneLine = b.ocrText.replacingOccurrences(of: "\n", with: " | ")
                            print(String(format: "frame %06d  %.1f ms  %@  score=%.3f  OCR: %@",
                                         frameIndex, dt * 1000.0, b.name, b.approachingScore, oneLine))
                        }
                    }
                }

            } catch {
                fputs("Frame \(frameIndex) error: \(error)\n", stderr)
            }
        }
    }

    let t1 = CFAbsoluteTimeGetCurrent()
    let total = t1 - t0
    let fps = (total > 0) ? Double(frames) / total : 0
    let avgProc = (frames > 0) ? procTimeSum / Double(frames) : 0
    let procFPS = (avgProc > 0) ? 1.0 / avgProc : 0

    // Summary JSON
    let summary: [String: Any] = [
        "video": videoURL.path,
        "dest": destURL.path,
        "framesProcessed": frames,
        "wallClockSeconds": total,
        "avgWallClockFPS": fps,
        "avgProcessingSecondsPerFrame": avgProc,
        "avgProcessingFPS": procFPS,
        "config": [
            "targetW": targetW,
            "targetH": targetH,
            "busClass": busClass,
            "s1conf": s1conf,
            "s2conf": s2conf,
            "preferLargestStage2": preferLargestStage2,
            "stage1CropMargin": stage1CropMargin,
            "stage2CropMargin": stage2CropMargin,
            "iou": iouThr,
            "maxMissed": maxMissed,
            "window": window,
            "minFrames": minFrames,
            "approach": approach
        ],
        "ocrPreset": [
            "scale": ocrScale,
            "flatten": ocrFlatten,
            "core": ocrCore,
            "binarize": ocrBinarize,
            "thresh": ocrThresh as Any,
            "shrink": ocrShrink,
            "thin": (args["ocrThin"] ?? "disabled"),
            "regrow": ocrRegrow
        ],
        "langs": recognitionLanguages ?? [],
        "frames": summaryFrames
    ]

    let summaryURL = destURL.appendingPathComponent("summary.json")
    try saveJSON(summary, to: summaryURL)

    print("\nDone.")
    print(String(format: "Frames: %d", frames))
    print(String(format: "Wall time: %.3fs  -> %.2f FPS", total, fps))
    print(String(format: "Avg processing: %.2f ms/frame -> %.2f FPS (processing-only)", avgProc * 1000.0, procFPS))
    print("Summary: \(summaryURL.path)")

} catch {
    usageAndExit("\(error)")
}
