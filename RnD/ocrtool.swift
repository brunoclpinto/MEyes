import Foundation
import Vision
import CoreImage
import ImageIO
import CoreGraphics

// MARK: - JSON output model

struct OCRResult: Codable {
    let file: String
    let text: String
}

// MARK: - OCR pipeline

final class OCRPipeline {
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Threshold kernel for binarization variants
    private let thresholdKernel: CIColorKernel? = CIColorKernel(source: """
    kernel vec4 thresholdFilter(__sample s, float t) {
        float l = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
        float v = step(t, l);
        return vec4(vec3(v), 1.0);
    }
    """)

    struct OCRPassResult {
        let text: String
        let avgConfidence: Float
    }

    func process(url: URL) -> String {
        guard let (cg, ori) = loadCGImage(url: url) else { return "" }

        // Oriented CIImage
        var base = CIImage(cgImage: cg).oriented(forExifOrientation: Int32(ori.rawValue))

        // Generic auto-scale to help OCR on tiny images but keep huge images manageable
        base = autoscale(base, targetMinDimension: 1200, maxDimension: 3000)

        // Build variants (NO CROPPING)
        let enhanced = preprocessEnhanced(base)

        // Binarized variants at different thresholds (helps across backgrounds)
        let thresholds: [CGFloat] = [0.45, 0.55, 0.65]
        let binVariants: [CIImage] = thresholds.compactMap { preprocessBinarized(enhanced, threshold: $0) }

        // OCR passes
        var candidates: [OCRPassResult] = []
        candidates.append(ocr(ciImage: base))
        candidates.append(ocr(ciImage: enhanced))
        for v in binVariants {
            candidates.append(ocr(ciImage: v))
        }

        // Pick best (digits heavily weighted)
        return pickBest(candidates).text
    }

    // MARK: - Load image

    private func loadCGImage(url: URL) -> (CGImage, CGImagePropertyOrientation)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            return nil
        }

        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let raw = (props?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        let ori = CGImagePropertyOrientation(rawValue: raw) ?? .up
        return (cg, ori)
    }

    // MARK: - Preprocessing

    private func autoscale(_ ci: CIImage, targetMinDimension: CGFloat, maxDimension: CGFloat) -> CIImage {
        let e = ci.extent.integral
        let w = e.width, h = e.height
        guard w > 0, h > 0 else { return ci }

        let minDim = min(w, h)
        let maxDim = max(w, h)

        var scale: CGFloat = 1.0

        // Upscale small images
        if minDim < targetMinDimension {
            scale = min(10.0, targetMinDimension / minDim)
        }

        // Downscale if it would get too large
        if maxDim * scale > maxDimension {
            scale = maxDimension / maxDim
        }

        guard abs(scale - 1.0) > 0.01 else { return ci }

        return ci.applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey: scale,
            kCIInputAspectRatioKey: 1.0
        ])
    }

    private func preprocessEnhanced(_ input: CIImage) -> CIImage {
        var ci = input

        // Grayscale + contrast
        ci = ci.applyingFilter("CIColorControls", parameters: [
            kCIInputSaturationKey: 0.0,
            kCIInputContrastKey: 2.2,
            kCIInputBrightnessKey: 0.0
        ])

        // Mild exposure + gamma to lift faint text
        ci = ci.applyingFilter("CIExposureAdjust", parameters: [
            kCIInputEVKey: 0.3
        ])
        ci = ci.applyingFilter("CIGammaAdjust", parameters: [
            "inputPower": 0.85
        ])

        // Denoise + sharpen
        ci = ci.applyingFilter("CINoiseReduction", parameters: [
            "inputNoiseLevel": 0.02,
            "inputSharpness": 0.5
        ])
        ci = ci.applyingFilter("CIUnsharpMask", parameters: [
            kCIInputRadiusKey: 2.0,
            kCIInputIntensityKey: 1.0
        ])

        return ci
    }

    private func preprocessBinarized(_ enhanced: CIImage, threshold: CGFloat) -> CIImage? {
        guard let k = thresholdKernel,
              let bin = k.apply(extent: enhanced.extent, arguments: [enhanced, threshold]) else {
            return nil
        }
        // Invert so text is dark on light (often helps OCR)
        return bin.applyingFilter("CIColorInvert")
    }

    // MARK: - Vision OCR

    private func ocr(ciImage: CIImage) -> OCRPassResult {
        guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return OCRPassResult(text: "", avgConfidence: 0)
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.recognitionLanguages = ["en-US"]
        request.minimumTextHeight = 0.01

        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return OCRPassResult(text: "", avgConfidence: 0)
        }

        let obs = request.results ?? []
        var lines: [String] = []
        var confSum: Float = 0
        var confCount: Float = 0

        for o in obs {
            guard let best = o.topCandidates(1).first else { continue }
            lines.append(best.string)
            confSum += best.confidence
            confCount += 1
        }

        let text = lines.joined(separator: "\n")
        let avg = confCount > 0 ? (confSum / confCount) : 0
        return OCRPassResult(text: text, avgConfidence: avg)
    }

    // MARK: - Choose best

    private func pickBest(_ candidates: [OCRPassResult]) -> OCRPassResult {
        func digitCount(_ s: String) -> Int { s.reduce(0) { $0 + ($1.isNumber ? 1 : 0) } }

        func score(_ r: OCRPassResult) -> Int {
            let digits = digitCount(r.text)
            let conf = Int(r.avgConfidence * 100.0)
            let len = r.text.count
            // digits dominate; then confidence; then text length
            return digits * 10000 + conf * 10 + len
        }

        return candidates.max(by: { score($0) < score($1) }) ?? OCRPassResult(text: "", avgConfidence: 0)
    }
}

// MARK: - CLI utilities

func stderr(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

func isDirectory(_ url: URL) -> Bool {
    var isDir: ObjCBool = false
    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
    return isDir.boolValue
}

func listJPGsRecursively(in folder: URL) -> [URL] {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: folder,
                                        includingPropertiesForKeys: [.isRegularFileKey],
                                        options: [.skipsHiddenFiles]) else {
        return []
    }
    var files: [URL] = []
    for case let url as URL in enumerator {
        let ext = url.pathExtension.lowercased()
        if ext == "jpg" || ext == "jpeg" {
            files.append(url)
        }
    }
    return files.sorted { $0.path < $1.path }
}

func relativePath(from base: URL, to file: URL) -> String {
    let basePath = base.standardizedFileURL.path
    let filePath = file.standardizedFileURL.path
    if filePath.hasPrefix(basePath) {
        var rel = String(filePath.dropFirst(basePath.count))
        if rel.hasPrefix("/") { rel.removeFirst() }
        return rel.isEmpty ? file.lastPathComponent : rel
    }
    return file.lastPathComponent
}

// MARK: - Main

let args = CommandLine.arguments
guard args.count >= 2 else {
    stderr("Usage: ocrtool <file.jpg|folder>")
    exit(1)
}

let inputURL = URL(fileURLWithPath: args[1])
let pipeline = OCRPipeline()

var results: [OCRResult] = []

if isDirectory(inputURL) {
    let files = listJPGsRecursively(in: inputURL)
    for f in files {
        let text = pipeline.process(url: f)
        results.append(OCRResult(file: relativePath(from: inputURL, to: f), text: text))
    }
} else {
    let ext = inputURL.pathExtension.lowercased()
    guard ext == "jpg" || ext == "jpeg" else {
        stderr("Input is not a .jpg/.jpeg: \(inputURL.path)")
        exit(2)
    }
    let text = pipeline.process(url: inputURL)
    results.append(OCRResult(file: inputURL.lastPathComponent, text: text))
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]

do {
    let data = try encoder.encode(results)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
} catch {
    stderr("Failed to encode JSON: \(error)")
    exit(3)
}

