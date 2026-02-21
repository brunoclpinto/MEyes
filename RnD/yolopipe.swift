import Foundation
import AVFoundation
import CoreML
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import Metal
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

// =====================
// CONFIG DEFAULTS
// =====================
let TARGET_W = 512
let TARGET_H = 896

let STAGE1_CLASS_ID_DEFAULT = 5
let S1_CONF_DEFAULT: Double = 0.25
let S2_CONF_DEFAULT: Double = 0.25
let MAX_S1_DEFAULT = 1000
let MAX_S2_DEFAULT = 1000

// =====================
// Errors
// =====================
enum AppError: Error, CustomStringConvertible {
    case badArgs(String)
    case fileNotFound(String)
    case cannotCreateDir(String)
    case noVideoTrack
    case readerFailed(String)
    case modelNoInput
    case modelNoOutput
    case modelOutputNotMultiArray(String)
    case cannotMakePixelBuffer
    case cannotMakeCGImage
    case cannotReadAttributes(String)

    var description: String {
        switch self {
        case .badArgs(let s): return "Bad args: \(s)"
        case .fileNotFound(let s): return "File not found: \(s)"
        case .cannotCreateDir(let s): return "Cannot create directory: \(s)"
        case .noVideoTrack: return "No video track found in the input file."
        case .readerFailed(let s): return "AVAssetReader failed: \(s)"
        case .modelNoInput: return "Model has no inputs."
        case .modelNoOutput: return "Model has no outputs."
        case .modelOutputNotMultiArray(let name): return "Model output '\(name)' is not an MLMultiArray."
        case .cannotMakePixelBuffer: return "Cannot create CVPixelBuffer."
        case .cannotMakeCGImage: return "Cannot create CGImage."
        case .cannotReadAttributes(let p): return "Cannot read file attributes: \(p)"
        }
    }
}

// =====================
// Arg parsing (name=value)
// =====================
func parseNameValueArgs(_ args: [String]) -> [String: String] {
    var dict: [String: String] = [:]
    for a in args.dropFirst() {
        if let eq = a.firstIndex(of: "=") {
            let name = String(a[..<eq])
            let value = String(a[a.index(after: eq)...])
            if !name.isEmpty { dict[name] = value }
        }
    }
    return dict
}

func require(_ dict: [String: String], _ key: String) throws -> String {
    guard let v = dict[key], !v.isEmpty else { throw AppError.badArgs("Missing \(key)=...") }
    return v
}

func optInt(_ dict: [String: String], _ key: String, _ def: Int) -> Int {
    if let s = dict[key], let v = Int(s) { return v }
    return def
}

func optDouble(_ dict: [String: String], _ key: String, _ def: Double) -> Double {
    if let s = dict[key], let v = Double(s) { return v }
    return def
}

func optFloat(_ dict: [String: String], _ key: String, _ def: Float) -> Float {
    if let s = dict[key], let v = Float(s) { return v }
    return def
}

func optBool(_ dict: [String: String], _ key: String, _ def: Bool) -> Bool {
    guard let s = dict[key]?.lowercased() else { return def }
    if ["1","true","yes","y","on"].contains(s) { return true }
    if ["0","false","no","n","off"].contains(s) { return false }
    return def
}

func optString(_ dict: [String: String], _ key: String, _ def: String? = nil) -> String? {
    if let s = dict[key], !s.isEmpty { return s }
    return def
}

func ensureFileExists(_ path: String) throws {
    if !FileManager.default.fileExists(atPath: path) { throw AppError.fileNotFound(path) }
}

func ensureDir(_ url: URL) throws {
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) {
        if isDir.boolValue { return }
        throw AppError.cannotCreateDir(url.path)
    }
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
}

// =====================
// Core Image / Metal context
// =====================
enum SharedCI {
    static let context: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()
}

// =====================
// Model compile helper (mlpackage/mlmodel -> mlmodelc)
// Caches into dest/_compiled_models/
// =====================
func fileModTime(_ url: URL) throws -> Date {
    do {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs[.modificationDate] as? Date) ?? Date.distantPast
    } catch {
        throw AppError.cannotReadAttributes(url.path)
    }
}

func compiledModelURL(for srcURL: URL, cacheDir: URL) throws -> URL {
    if srcURL.pathExtension == "mlmodelc" { return srcURL }
    try ensureDir(cacheDir)

    let base = srcURL.deletingPathExtension().lastPathComponent
    let cached = cacheDir.appendingPathComponent(base).appendingPathExtension("mlmodelc")

    if FileManager.default.fileExists(atPath: cached.path) {
        let srcTime = try fileModTime(srcURL)
        let cachedTime = try fileModTime(cached)
        if cachedTime >= srcTime { return cached }
        try? FileManager.default.removeItem(at: cached)
    }

    let t0 = CFAbsoluteTimeGetCurrent()
    let compiledTemp = try MLModel.compileModel(at: srcURL)
    let t1 = CFAbsoluteTimeGetCurrent()
    print(String(format: "Compiled %@ in %.3fs", srcURL.lastPathComponent, (t1 - t0)))

    try? FileManager.default.removeItem(at: cached)
    try FileManager.default.copyItem(at: compiledTemp, to: cached)
    return cached
}

// =====================
// Image helpers
// =====================
func letterbox(_ image: CIImage, targetW: Int, targetH: Int) -> CIImage {
    let src = image.extent.integral
    let dstRect = CGRect(x: 0, y: 0, width: targetW, height: targetH)

    let sx = CGFloat(targetW) / max(src.width, 1)
    let sy = CGFloat(targetH) / max(src.height, 1)
    let scale = min(sx, sy)

    let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let se = scaled.extent.integral

    let dx = (dstRect.width - se.width) / 2.0 - se.origin.x
    let dy = (dstRect.height - se.height) / 2.0 - se.origin.y
    let translated = scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))

    let bg = CIImage(color: .black).cropped(to: dstRect)
    return translated.composited(over: bg).cropped(to: dstRect)
}

func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)
    ]
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                    kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    guard status == kCVReturnSuccess, let out = pb else { throw AppError.cannotMakePixelBuffer }
    return out
}

func renderToPixelBuffer(_ image: CIImage, pb: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let rect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
    SharedCI.context.render(image, to: pb, bounds: rect, colorSpace: CGColorSpaceCreateDeviceRGB())
}

func renderToCGImage(_ image: CIImage) throws -> CGImage {
    let rect = image.extent.integral
    guard let cg = SharedCI.context.createCGImage(image, from: rect) else { throw AppError.cannotMakeCGImage }
    return cg
}

func savePNG(_ cgImage: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw AppError.cannotMakeCGImage
    }
    CGImageDestinationAddImage(dest, cgImage, nil)
    CGImageDestinationFinalize(dest)
}

/// Convert YOLO box (assumed top-left origin) to CIImage crop rect (CI origin bottom-left)
func ciCropRectFromTopLeftXYXY(x1: Double, y1: Double, x2: Double, y2: Double, imgW: Double, imgH: Double) -> CGRect {
    let left = max(0.0, min(imgW, x1))
    let right = max(0.0, min(imgW, x2))
    let top = max(0.0, min(imgH, y1))
    let bottom = max(0.0, min(imgH, y2))

    let w = max(0.0, right - left)
    let h = max(0.0, bottom - top)

    // top-left -> CI bottom-left
    let ciY = imgH - bottom
    return CGRect(x: left, y: ciY, width: w, height: h).integral
}

func clampRect(_ r: CGRect, to bounds: CGRect) -> CGRect {
    let rr = r.intersection(bounds)
    return rr.isNull ? .zero : rr.integral
}

// =====================
// CoreML YOLO wrapper
// =====================
final class YOLOCoreML {
    let model: MLModel
    let inputName: String
    let outputName: String

    init(compiledURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        self.model = try MLModel(contentsOf: compiledURL, configuration: cfg)

        let inputs = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard !inputs.isEmpty else { throw AppError.modelNoInput }
        self.inputName = inputs.contains("image") ? "image" : inputs[0]

        let outputs = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard !outputs.isEmpty else { throw AppError.modelNoOutput }
        self.outputName = outputs[0]
    }

    func predict(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let out = try model.prediction(from: provider)
        guard let fv = out.featureValue(for: outputName), let arr = fv.multiArrayValue else {
            throw AppError.modelOutputNotMultiArray(outputName)
        }
        return arr
    }
}

// =====================
// YOLO output parsing (Nx6-ish)
// =====================
struct Detection {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double
    var score: Double
    var cls: Int
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
            f = frac == 0 ? Float.infinity : Float.nan
        } else {
            f = (1.0 + Float(frac) / Float(1 << 10)) * powf(2, Float(exp - 15))
        }
        return Double(sign ? -f : f)
    default:
        return Double(truncating: arr[linearIndex] as NSNumber)
    }
}

func parseDetections(_ arr: MLMultiArray, inputW: Double, inputH: Double) -> [Detection] {
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

        // xyxy vs cxcywh heuristic
        if c <= a || d <= b {
            let cx = a, cy = b, w = c, h = d
            a = cx - w/2.0
            b = cy - h/2.0
            c = cx + w/2.0
            d = cy + h/2.0
        }

        out.append(Detection(x1: a, y1: b, x2: c, y2: d, score: score, cls: cls))
    }
    return out
}

// =====================
// OCR (CGImage-based adaptation of your improved pipeline)
// =====================
struct OCRParams: Sendable {
    enum Thin: Sendable {
        case disabled
        case enabled(maxIterations: Int = 0) // 0 = until stable
    }

    var scale: Float
    var flatten: Bool
    var core: Float
    var binarize: Bool
    var thresh: Float?   // nil => Otsu when binarize=true
    var shrink: Float
    var thin: Thin
    var regrow: Float

    init(
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
}

enum OCRError: Error {
    case cannotRender
    case visionFailed(Error)
}

private enum OCRPipeline {
    static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    static func renderToCGImage(_ img: CIImage) throws -> CGImage {
        guard let cg = ciContext.createCGImage(img, from: img.extent.integral) else {
            throw OCRError.cannotRender
        }
        return cg
    }

    static func visionOCR(
        cgImage: CGImage,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = recognitionLevel
        request.usesLanguageCorrection = usesLanguageCorrection
        if let langs = recognitionLanguages, !langs.isEmpty { request.recognitionLanguages = langs }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let obs = request.results ?? []
        let sorted = obs.sorted {
            if $0.boundingBox.minY != $1.boundingBox.minY { return $0.boundingBox.minY > $1.boundingBox.minY }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }

        return sorted.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }

    // FIXED: no overlapping access to `px`
    static func meanBrightness(_ image: CIImage) -> Double {
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

    static func cgImageToGrayBytes(_ cg: CGImage) -> (bytes: [UInt8], width: Int, height: Int)? {
        let w = cg.width
        let h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h)

        let ok: Bool = buf.withUnsafeMutableBytes { ptr in
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

    static func grayBytesToCGImage(_ bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        var data = bytes
        guard let provider = CGDataProvider(data: Data(bytes: &data, count: data.count) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    // ---- Thinning (Zhang–Suen)
    static func thinBinaryCIImage(_ img: CIImage, maxIterations: Int) throws -> CIImage {
        let cg = try renderToCGImage(img)
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

    static func zhangSuenThin(_ src: [UInt8], width: Int, height: Int, maxIterations: Int) -> [UInt8] {
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
}

func ocrCGImage(
    cgInput: CGImage,
    params: OCRParams,
    recognitionLanguages: [String]?,
    usesLanguageCorrection: Bool,
    recognitionLevel: VNRequestTextRecognitionLevel
) async throws -> (text: String, preprocessed: CGImage) {

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
                if params.flatten {
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

                // 3) upscale (BEFORE any binarization)
                if params.scale != 1 {
                    let lz = CIFilter.lanczosScaleTransform()
                    lz.inputImage = img
                    lz.scale = params.scale
                    lz.aspectRatio = 1.0
                    let out = lz.outputImage ?? img
                    img = out.cropped(to: out.extent.integral)
                }

                // 4) grayscale "core tighten" (pre-threshold)
                if params.core >= 1 {
                    let core = CIFilter.morphologyRectangleMaximum()
                    core.inputImage = img
                    core.width = params.core
                    core.height = params.core
                    let out = core.outputImage ?? img
                    img = out.cropped(to: out.extent.integral)
                }

                // If binarize=false -> grayscale to Vision
                if !params.binarize {
                    let cgOut = try OCRPipeline.renderToCGImage(img)
                    let text = try OCRPipeline.visionOCR(
                        cgImage: cgOut,
                        recognitionLanguages: recognitionLanguages,
                        usesLanguageCorrection: usesLanguageCorrection,
                        recognitionLevel: recognitionLevel
                    )
                    continuation.resume(returning: (text, cgOut))
                    return
                }

                // 5) threshold (post-upscale, post-core)
                if let t = params.thresh {
                    let tt = max(0, min(1, t))
                    img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": tt])
                        .cropped(to: img.extent.integral)
                } else {
                    if #available(macOS 11.0, *) {
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
                if OCRPipeline.meanBrightness(img) < 0.5 {
                    img = img.applyingFilter("CIColorInvert").cropped(to: img.extent.integral)
                }

                // 6) optional shrink (post-threshold)
                if params.shrink >= 1 {
                    let shrink = CIFilter.morphologyRectangleMaximum() // white expands => black contracts
                    shrink.inputImage = img
                    shrink.width = params.shrink
                    shrink.height = params.shrink
                    let out = shrink.outputImage ?? img
                    img = out.cropped(to: out.extent.integral)
                }

                // 7) optional thinning (post-threshold)
                switch params.thin {
                case .disabled:
                    break
                case .enabled(let maxIter):
                    img = try OCRPipeline.thinBinaryCIImage(img, maxIterations: max(0, maxIter))
                }

                // 8) optional regrow (post-threshold)
                if params.regrow >= 1 {
                    let grow = CIFilter.morphologyRectangleMinimum() // black expands
                    grow.inputImage = img
                    grow.width = params.regrow
                    grow.height = params.regrow
                    let out = grow.outputImage ?? img
                    img = out.cropped(to: out.extent.integral)
                }

                let cgOut = try OCRPipeline.renderToCGImage(img)
                let text = try OCRPipeline.visionOCR(
                    cgImage: cgOut,
                    recognitionLanguages: recognitionLanguages,
                    usesLanguageCorrection: usesLanguageCorrection,
                    recognitionLevel: recognitionLevel
                )
                continuation.resume(returning: (text, cgOut))

            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

func extractNaturalNumbers(_ text: String) -> [String] {
    let parts = text.components(separatedBy: CharacterSet.decimalDigits.inverted)
    return parts.filter { !$0.isEmpty }
}

// =====================
// JSON output
// =====================
struct OCRRecord: Codable {
    let frameIndex: Int
    let stage1Index: Int
    let stage2Index: Int
    let stage1Crop: String
    let stage2Crop: String
    let preprocessed: String
    let stage1Score: Double
    let stage2Score: Double
    let stage2Class: Int
    let ocrRaw: String
    let naturalNumbers: [String]
    let bestNumber: String?
}

// =====================
// Async run
// =====================
func runAsync() async throws {
    let args = parseNameValueArgs(CommandLine.arguments)

    let videoPath = try require(args, "video")
    let destPath = try require(args, "dest")
    let yoloPath = try require(args, "yolo")
    let yoloCustomPath = try require(args, "yoloCustom")

    try ensureFileExists(videoPath)
    try ensureFileExists(yoloPath)
    try ensureFileExists(yoloCustomPath)

    let maxFrames = args["maxFrames"].flatMap(Int.init)
    let s1conf = optDouble(args, "s1conf", S1_CONF_DEFAULT)
    let s2conf = optDouble(args, "s2conf", S2_CONF_DEFAULT)
    let maxS1 = optInt(args, "maxS1", MAX_S1_DEFAULT)
    let maxS2 = optInt(args, "maxS2", MAX_S2_DEFAULT)
    let stage1ClassId = optInt(args, "s1class", STAGE1_CLASS_ID_DEFAULT)

    // OCR params (defaults = your starting preset)
    let ocrScale = optFloat(args, "ocrScale", 1)
    let ocrFlatten = optBool(args, "ocrFlatten", true)
    let ocrCore = optFloat(args, "ocrCore", 1)
    let ocrBinarize = optBool(args, "ocrBinarize", false)
    let ocrThresh: Float? = {
        guard let s = optString(args, "ocrThresh") else { return nil }
        return Float(s)
    }()
    let ocrShrink = optFloat(args, "ocrShrink", 0)
    let ocrRegrow = optFloat(args, "ocrRegrow", 0)

    let thinMode = (optString(args, "ocrThin", "disabled") ?? "disabled").lowercased()
    let thinMaxIter = optInt(args, "ocrThinMaxIter", 0)
    let ocrThin: OCRParams.Thin = (thinMode == "enabled") ? .enabled(maxIterations: thinMaxIter) : .disabled

    let ocrParams = OCRParams(
        scale: ocrScale,
        flatten: ocrFlatten,
        core: ocrCore,
        binarize: ocrBinarize,
        thresh: ocrThresh,
        shrink: ocrShrink,
        thin: ocrThin,
        regrow: ocrRegrow
    )

    let langs: [String]? = {
        guard let s = optString(args, "ocrLangs") else { return nil }
        let parts = s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        return parts.isEmpty ? nil : parts
    }()
    let ocrLangCorrection = optBool(args, "ocrLangCorrection", false)
    let levelStr = (optString(args, "ocrLevel", "accurate") ?? "accurate").lowercased()
    let ocrLevel: VNRequestTextRecognitionLevel = (levelStr == "fast") ? .fast : .accurate

    let videoURL = URL(fileURLWithPath: videoPath)
    let destURL = URL(fileURLWithPath: destPath)

    let step1Dir = destURL.appendingPathComponent("step1_crops", isDirectory: true)
    let step2Dir = destURL.appendingPathComponent("step2_crops", isDirectory: true)
    let prepDir  = destURL.appendingPathComponent("preprocessed", isDirectory: true)
    let compiledDir = destURL.appendingPathComponent("_compiled_models", isDirectory: true)

    try ensureDir(destURL)
    try ensureDir(step1Dir)
    try ensureDir(step2Dir)
    try ensureDir(prepDir)
    try ensureDir(compiledDir)

    print("Compiling/loading models...")
    let tModel0 = CFAbsoluteTimeGetCurrent()

    let yoloCompiled = try compiledModelURL(for: URL(fileURLWithPath: yoloPath), cacheDir: compiledDir)
    let yoloCustomCompiled = try compiledModelURL(for: URL(fileURLWithPath: yoloCustomPath), cacheDir: compiledDir)

    let model1 = try YOLOCoreML(compiledURL: yoloCompiled)
    let model2 = try YOLOCoreML(compiledURL: yoloCustomCompiled)

    let tModel1 = CFAbsoluteTimeGetCurrent()
    print(String(format: "Models ready in %.3fs", (tModel1 - tModel0)))

    print("OCR params: scale=\(ocrParams.scale) flatten=\(ocrParams.flatten) core=\(ocrParams.core) binarize=\(ocrParams.binarize) thresh=\(String(describing: ocrParams.thresh)) shrink=\(ocrParams.shrink) thin=\(thinMode)(\(thinMaxIter)) regrow=\(ocrParams.regrow) level=\(levelStr) langs=\(langs?.joined(separator: ",") ?? "-")")

    // NEW: async track + preferredTransform loading (no deprecation warnings)
    let asset = AVURLAsset(url: videoURL)
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard let track = tracks.first else { throw AppError.noVideoTrack }
    let preferredTransform = try await track.load(.preferredTransform)

    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ])
    output.alwaysCopiesSampleData = false

    guard reader.canAdd(output) else { throw AppError.readerFailed("Cannot add track output.") }
    reader.add(output)
    guard reader.startReading() else {
        throw AppError.readerFailed(reader.error?.localizedDescription ?? "Unknown error")
    }

    var records: [OCRRecord] = []
    var frameIndex = 0
    var totalFramesRead = 0
    var totalStage1Crops = 0
    var totalStage2Crops = 0

    let overallStart = CFAbsoluteTimeGetCurrent()

    while let sbuf = output.copyNextSampleBuffer() {
        if let mf = maxFrames, frameIndex >= mf { break }
        totalFramesRead += 1

        guard let pb = CMSampleBufferGetImageBuffer(sbuf) else {
            frameIndex += 1
            continue
        }

        // Apply preferredTransform to get upright frames (important for iPhone videos)
        var frameCI = CIImage(cvPixelBuffer: pb).transformed(by: preferredTransform)
        // normalize origin to (0,0)
        let e = frameCI.extent
        frameCI = frameCI.transformed(by: CGAffineTransform(translationX: -e.origin.x, y: -e.origin.y))

        // Step 1: pad/letterbox to 512x896
        let stage1InputCI = letterbox(frameCI, targetW: TARGET_W, targetH: TARGET_H)
        let stage1PB = try makePixelBuffer(width: TARGET_W, height: TARGET_H)
        renderToPixelBuffer(stage1InputCI, pb: stage1PB)

        // YOLO #1 (class 5)
        let out1 = try model1.predict(pixelBuffer: stage1PB)
        var det1 = parseDetections(out1, inputW: Double(TARGET_W), inputH: Double(TARGET_H))
        det1 = det1
            .filter { $0.cls == stage1ClassId && $0.score >= s1conf }
            .sorted { $0.score > $1.score }
        if det1.count > maxS1 { det1 = Array(det1.prefix(maxS1)) }

        if det1.isEmpty {
            frameIndex += 1
            if frameIndex % 30 == 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - overallStart
                let fps = elapsed > 0 ? Double(totalFramesRead) / elapsed : 0
                print(String(format: "Frame %d | elapsed %.2fs | FPS %.2f | no stage1 detections",
                             frameIndex, elapsed, fps))
            }
            continue
        }

        for (s1i, d1) in det1.enumerated() {
            var r1 = ciCropRectFromTopLeftXYXY(
                x1: d1.x1, y1: d1.y1, x2: d1.x2, y2: d1.y2,
                imgW: Double(TARGET_W), imgH: Double(TARGET_H)
            )
            r1 = clampRect(r1, to: CGRect(x: 0, y: 0, width: TARGET_W, height: TARGET_H))
            if r1.width < 2 || r1.height < 2 { continue }

            // Crop stage1
            let crop1CI = stage1InputCI.cropped(to: r1)
            let crop1CG = try renderToCGImage(crop1CI)

            let crop1Name = String(format: "f_%06d_s1_%02d.png", frameIndex, s1i)
            try savePNG(crop1CG, to: step1Dir.appendingPathComponent(crop1Name))
            totalStage1Crops += 1

            // Pad crop to 512x896
            let stage2InputCI = letterbox(crop1CI, targetW: TARGET_W, targetH: TARGET_H)
            let stage2PB = try makePixelBuffer(width: TARGET_W, height: TARGET_H)
            renderToPixelBuffer(stage2InputCI, pb: stage2PB)

            // YOLO #2 custom
            let out2 = try model2.predict(pixelBuffer: stage2PB)
            var det2 = parseDetections(out2, inputW: Double(TARGET_W), inputH: Double(TARGET_H))
            det2 = det2
                .filter { $0.score >= s2conf }
                .sorted { $0.score > $1.score }
            if det2.count > maxS2 { det2 = Array(det2.prefix(maxS2)) }
            if det2.isEmpty { continue }

            for (s2i, d2) in det2.enumerated() {
                var r2 = ciCropRectFromTopLeftXYXY(
                    x1: d2.x1, y1: d2.y1, x2: d2.x2, y2: d2.y2,
                    imgW: Double(TARGET_W), imgH: Double(TARGET_H)
                )
                r2 = clampRect(r2, to: CGRect(x: 0, y: 0, width: TARGET_W, height: TARGET_H))
                if r2.width < 2 || r2.height < 2 { continue }

                // Crop stage2
                let crop2CI = stage2InputCI.cropped(to: r2)
                let crop2CG = try renderToCGImage(crop2CI)

                let crop2Name = String(format: "f_%06d_s1_%02d_s2_%02d.png", frameIndex, s1i, s2i)
                try savePNG(crop2CG, to: step2Dir.appendingPathComponent(crop2Name))
                totalStage2Crops += 1

                // OCR
                let (ocrRaw, preCG) = try await ocrCGImage(
                    cgInput: crop2CG,
                    params: ocrParams,
                    recognitionLanguages: langs,
                    usesLanguageCorrection: ocrLangCorrection,
                    recognitionLevel: ocrLevel
                )

                let nums = extractNaturalNumbers(ocrRaw)
                let best = nums.first

                let prepName = String(format: "f_%06d_s1_%02d_s2_%02d_prep.png", frameIndex, s1i, s2i)
                try savePNG(preCG, to: prepDir.appendingPathComponent(prepName))

                records.append(OCRRecord(
                    frameIndex: frameIndex,
                    stage1Index: s1i,
                    stage2Index: s2i,
                    stage1Crop: "step1_crops/\(crop1Name)",
                    stage2Crop: "step2_crops/\(crop2Name)",
                    preprocessed: "preprocessed/\(prepName)",
                    stage1Score: d1.score,
                    stage2Score: d2.score,
                    stage2Class: d2.cls,
                    ocrRaw: ocrRaw,
                    naturalNumbers: nums,
                    bestNumber: best
                ))

                // Terminal output: natural numbers only
                if !nums.isEmpty {
                    print("OCR nums: \(nums.joined(separator: " ")) | file=\(crop2Name)")
                }
            }
        }

        frameIndex += 1

        if frameIndex % 30 == 0 {
            let elapsed = CFAbsoluteTimeGetCurrent() - overallStart
            let fps = elapsed > 0 ? Double(totalFramesRead) / elapsed : 0
            print(String(format: "Frame %d | elapsed %.2fs | FPS %.2f | s1crops=%d s2crops=%d",
                         frameIndex, elapsed, fps, totalStage1Crops, totalStage2Crops))
        }
    }

    let totalTime = CFAbsoluteTimeGetCurrent() - overallStart
    let fps = totalTime > 0 ? Double(totalFramesRead) / totalTime : 0

    // Write JSON
    let jsonURL = destURL.appendingPathComponent("results.json")
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try enc.encode(records)
    try data.write(to: jsonURL, options: [.atomic])

    print("---- DONE ----")
    print("Frames read: \(totalFramesRead)")
    print("Stage1 crops saved: \(totalStage1Crops)")
    print("Stage2 crops saved: \(totalStage2Crops)")
    print(String(format: "Processing time: %.3fs", totalTime))
    print(String(format: "FPS: %.2f", fps))
    print("JSON: \(jsonURL.path)")
}

// =====================
// Script entry (no @main)
// =====================
let sem = DispatchSemaphore(value: 0)
Task {
    do {
        try await runAsync()
        sem.signal()
    } catch {
        fputs("Error: \(error)\n", stderr)
        sem.signal()
        exit(1)
    }
}
sem.wait()

