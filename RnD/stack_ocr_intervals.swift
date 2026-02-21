import Foundation
import Vision
import CoreImage
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

// MARK: - Filename parsing: last digit run before .jpg/.jpeg

func parseFrameNumber(filename: String) -> Int? {
    // Extract LAST contiguous run of digits right before extension.
    // Examples:
    //   IMG_9180_99.jpg  -> 99
    //   BusInfo000123.jpg -> 123
    //   label-0007.jpeg  -> 7
    let lower = filename.lowercased()
    guard lower.hasSuffix(".jpg") || lower.hasSuffix(".jpeg") else { return nil }

    let base: String
    if lower.hasSuffix(".jpeg") {
        base = String(filename.dropLast(5))
    } else {
        base = String(filename.dropLast(4))
    }

    var digits = ""
    for ch in base.reversed() {
        if ch.isNumber { digits.append(ch) }
        else { break }
    }
    guard !digits.isEmpty else { return nil }
    return Int(String(digits.reversed()))
}

// MARK: - CLI

func usage() -> Never {
    print("""
    Usage:
      stack_ocr_intervals <folder>
        [--fps 30]               (defaults to 30 if omitted)
        [--maxframe 200]         (ignore frames > maxframe)
        [--onebased]             (if your numbering starts at 1)
        [--lang pt-PT,en-US]
        [--stack 21]             (#frames to stack, odd recommended)
        [--step 1]               (process every Nth frame; default 1)
        [--minchars 3]           (minimum chars to count as "text detected")
        [--minconf 0.50]         (minimum best OCR confidence)
        [--sharpen 0.6] [--radius 2.0] [--contrast 1.15]
        [--debugout <folder>]    (save stacked images for inspection)
        [--startswith <prefix>]  (optional: only files starting with prefix)

    Examples:
      stack_ocr_intervals .
      stack_ocr_intervals . --fps 30
      stack_ocr_intervals . --stack 31 --minconf 0.40 --contrast 1.25
      stack_ocr_intervals . --debugout ./dbg
      stack_ocr_intervals . --startswith IMG_9180_
    """)
    exit(2)
}

let argv = Array(CommandLine.arguments.dropFirst())
guard !argv.isEmpty else { usage() }

var folderPath: String? = nil
var fps: Double = 30.0
var maxFrame: Int = 200
var oneBased = false
var languages = ["en-US", "pt-PT"]

var stackCount = 21
var step = 1

var minChars = 3
var minConf: Float = 0.50

var sharpen: Double = 0.6
var radius: Double = 2.0
var contrast: Double = 1.15

var debugOut: URL? = nil
var startsWith: String? = nil

var i = 0
while i < argv.count {
    let a = argv[i]
    switch a {
    case "--fps":
        guard i+1 < argv.count, let v = Double(argv[i+1]), v > 0 else { usage() }
        fps = v; i += 2
    case "--maxframe":
        guard i+1 < argv.count, let v = Int(argv[i+1]), v >= 0 else { usage() }
        maxFrame = v; i += 2
    case "--onebased":
        oneBased = true; i += 1
    case "--lang":
        guard i+1 < argv.count else { usage() }
        languages = argv[i+1].split(separator: ",").map(String.init)
        i += 2
    case "--stack":
        guard i+1 < argv.count, let v = Int(argv[i+1]), v >= 3 else { usage() }
        stackCount = v; i += 2
    case "--step":
        guard i+1 < argv.count, let v = Int(argv[i+1]), v >= 1 else { usage() }
        step = v; i += 2
    case "--minchars":
        guard i+1 < argv.count, let v = Int(argv[i+1]), v >= 0 else { usage() }
        minChars = v; i += 2
    case "--minconf":
        guard i+1 < argv.count, let v = Float(argv[i+1]) else { usage() }
        minConf = v; i += 2
    case "--sharpen":
        guard i+1 < argv.count, let v = Double(argv[i+1]) else { usage() }
        sharpen = v; i += 2
    case "--radius":
        guard i+1 < argv.count, let v = Double(argv[i+1]) else { usage() }
        radius = v; i += 2
    case "--contrast":
        guard i+1 < argv.count, let v = Double(argv[i+1]) else { usage() }
        contrast = v; i += 2
    case "--debugout":
        guard i+1 < argv.count else { usage() }
        debugOut = URL(fileURLWithPath: argv[i+1])
        i += 2
    case "--startswith":
        guard i+1 < argv.count else { usage() }
        startsWith = argv[i+1]
        i += 2
    default:
        if folderPath == nil { folderPath = a; i += 1 }
        else { usage() }
    }
}

guard let folder = folderPath else { usage() }

// MARK: - Image helpers

func ensureDir(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
}

func savePNG(_ cgImage: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "save", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create image destination"])
    }
    CGImageDestinationAddImage(dest, cgImage, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "save", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot finalize image destination"])
    }
}

// Pad crop to common canvas centered + produce valid-pixel mask
func paddedToCanvas(_ img: CIImage, canvasRect: CGRect) -> (padded: CIImage, mask: CIImage) {
    let extent = img.extent.integral
    let bg = CIImage(color: .black).cropped(to: canvasRect)

    let x = (canvasRect.width - extent.width) * 0.5
    let y = (canvasRect.height - extent.height) * 0.5

    let t = CGAffineTransform(translationX: x - extent.origin.x, y: y - extent.origin.y)
    let moved = img.transformed(by: t)

    let whiteRect = CIImage(color: .white).cropped(to: CGRect(x: x, y: y, width: extent.width, height: extent.height))
    let mask = whiteRect.composited(over: bg).cropped(to: canvasRect)

    let padded = moved.composited(over: bg).cropped(to: canvasRect)
    return (padded, mask)
}

// Render CIImage -> grayscale R8 buffer
final class GrayRenderer {
    let context: CIContext
    let grayCS = CGColorSpaceCreateDeviceGray()

    init() {
        self.context = CIContext(options: [.cacheIntermediates: false])
    }

    func renderR8(_ image: CIImage, rect: CGRect, width: Int, height: Int) -> [UInt8] {
        var buf = [UInt8](repeating: 0, count: width * height)
        buf.withUnsafeMutableBytes { ptr in
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: width,
                bounds: rect,
                format: .R8,
                colorSpace: grayCS
            )
        }
        return buf
    }

    func makeGrayCGImage(from buf: [UInt8], width: Int, height: Int) -> CGImage? {
        let cfData = CFDataCreate(nil, buf, buf.count)
        guard let provider = CGDataProvider(data: cfData!) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: grayCS,
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}

// Vision translation alignment: align moving -> reference
func alignToReference(
    moving: CIImage,
    movingMask: CIImage,
    reference: CIImage,
    canvasRect: CGRect
) -> (CIImage, CIImage)? {
    let request = VNTranslationalImageRegistrationRequest(targetedCIImage: moving, options: [:])
    let handler = VNImageRequestHandler(ciImage: reference, options: [:])

    do {
        try handler.perform([request])
        guard let obs = request.results?.first as? VNImageTranslationAlignmentObservation else { return nil }
        let t = obs.alignmentTransform
        let aligned = moving.transformed(by: t).cropped(to: canvasRect)
        let alignedMask = movingMask.transformed(by: t).cropped(to: canvasRect)
        return (aligned, alignedMask)
    } catch {
        return nil
    }
}

// Median stack with mask (robust temporal denoise)
func medianStack(buffers: [[UInt8]], masks: [[UInt8]], width: Int, height: Int) -> [UInt8] {
    let n = buffers.count
    let total = width * height
    var out = [UInt8](repeating: 0, count: total)
    var tmp = [UInt8]()
    tmp.reserveCapacity(n)

    for p in 0..<total {
        tmp.removeAll(keepingCapacity: true)
        for k in 0..<n {
            if masks[k][p] > 0 {
                tmp.append(buffers[k][p])
            }
        }
        if tmp.isEmpty {
            out[p] = 0
        } else {
            tmp.sort()
            out[p] = tmp[tmp.count / 2]
        }
    }
    return out
}

// OCR
struct OCRResult {
    let text: String
    let bestConfidence: Float
}

func ocrCGImage(_ cgImage: CGImage, languages: [String]) throws -> OCRResult {
    var lines: [String] = []
    var bestConf: Float = 0

    let request = VNRecognizeTextRequest { req, err in
        if let err = err {
            fputs("OCR error: \(err)\n", stderr)
            return
        }
        guard let obs = req.results as? [VNRecognizedTextObservation] else { return }
        for o in obs {
            if let best = o.topCandidates(1).first {
                lines.append(best.string)
                if best.confidence > bestConf { bestConf = best.confidence }
            }
        }
    }

    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = languages

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])

    let text = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    return OCRResult(text: text, bestConfidence: bestConf)
}

func collapseWhitespace(_ s: String) -> String {
    let replaced = s.replacingOccurrences(of: "\n", with: " ")
    return replaced.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

// Time conversion
func frameToSecondsStart(_ frame: Int, fps: Double, oneBased: Bool) -> Double {
    let f = oneBased ? (frame - 1) : frame
    return Double(max(f, 0)) / fps
}
func frameToSecondsEndExclusive(_ frameInclusive: Int, fps: Double, oneBased: Bool) -> Double {
    // end covers the frame duration: (last+1)/fps (0-based), last/fps (1-based)
    let f = oneBased ? frameInclusive : (frameInclusive + 1)
    return Double(max(f, 0)) / fps
}

// MARK: - Load frames

let fm = FileManager.default
let dirURL = URL(fileURLWithPath: folder)

if let out = debugOut {
    try ensureDir(out)
}

let allFiles = try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])

var frameURLs: [(frame: Int, url: URL)] = allFiles.compactMap { url in
    let name = url.lastPathComponent
    if let sw = startsWith, !name.hasPrefix(sw) { return nil }
    guard let f = parseFrameNumber(filename: name), f <= maxFrame else { return nil }
    return (f, url)
}.sorted { $0.frame < $1.frame }

if frameURLs.isEmpty {
    fputs("No matching .jpg/.jpeg files with trailing frame number (<= \(maxFrame)) found in: \(dirURL.path)\n", stderr)
    exit(1)
}

// Load CIImages and compute max canvas size
var ciFrames: [(frame: Int, image: CIImage)] = []
ciFrames.reserveCapacity(frameURLs.count)

var maxW = 0
var maxH = 0

for item in frameURLs {
    if let img = CIImage(contentsOf: item.url) {
        let e = img.extent.integral
        maxW = max(maxW, Int(e.width))
        maxH = max(maxH, Int(e.height))
        ciFrames.append((item.frame, img))
    }
}

if ciFrames.isEmpty {
    fputs("Could not load any images as CIImage.\n", stderr)
    exit(1)
}

let canvasRect = CGRect(x: 0, y: 0, width: maxW, height: maxH)

let renderer = GrayRenderer()
let ciContext = renderer.context

// MARK: - Process frames: stack -> OCR -> per-frame results

struct FrameOCR {
    let frame: Int
    let ok: Bool
    let text: String
    let conf: Float
}

var results: [FrameOCR] = []
results.reserveCapacity((ciFrames.count + step - 1) / step)

let half = max(1, stackCount / 2)

for idx in stride(from: 0, to: ciFrames.count, by: step) {
    autoreleasepool {
        let (centerFrame, centerImg) = ciFrames[idx]
        let (refPadded, refMask) = paddedToCanvas(centerImg, canvasRect: canvasRect)

        var imgBuffers: [[UInt8]] = []
        var maskBuffers: [[UInt8]] = []
        imgBuffers.reserveCapacity(stackCount)
        maskBuffers.reserveCapacity(stackCount)

        // Include reference
        let refBuf = renderer.renderR8(refPadded, rect: canvasRect, width: maxW, height: maxH)
        let refMBuf = renderer.renderR8(refMask, rect: canvasRect, width: maxW, height: maxH)
        imgBuffers.append(refBuf)
        maskBuffers.append(refMBuf)

        let start = max(0, idx - half)
        let end = min(ciFrames.count - 1, idx + half)

        if start <= end {
            for j in start...end {
                if j == idx { continue }
                let (_, img) = ciFrames[j]
                let (movPadded, movMask) = paddedToCanvas(img, canvasRect: canvasRect)

                if let (aligned, alignedMask) = alignToReference(
                    moving: movPadded,
                    movingMask: movMask,
                    reference: refPadded,
                    canvasRect: canvasRect
                ) {
                    imgBuffers.append(renderer.renderR8(aligned, rect: canvasRect, width: maxW, height: maxH))
                    maskBuffers.append(renderer.renderR8(alignedMask, rect: canvasRect, width: maxW, height: maxH))
                }
            }
        }

        // Stack (median) if we have enough aligned frames
        let stackedBuf: [UInt8] = (imgBuffers.count >= 3)
            ? medianStack(buffers: imgBuffers, masks: maskBuffers, width: maxW, height: maxH)
            : refBuf

        guard let stackedCG = renderer.makeGrayCGImage(from: stackedBuf, width: maxW, height: maxH) else {
            results.append(FrameOCR(frame: centerFrame, ok: false, text: "", conf: 0))
            return
        }

        // Optional enhancement (unsharp + contrast)
        var outCI = CIImage(cgImage: stackedCG).cropped(to: canvasRect)

        if sharpen > 0 {
            outCI = outCI.applyingFilter("CIUnsharpMask", parameters: [
                kCIInputRadiusKey: radius,
                kCIInputIntensityKey: sharpen
            ])
        }

        if contrast != 1.0 {
            outCI = outCI.applyingFilter("CIColorControls", parameters: [
                kCIInputContrastKey: contrast
            ])
        }

        guard let finalCG = ciContext.createCGImage(outCI, from: canvasRect) else {
            results.append(FrameOCR(frame: centerFrame, ok: false, text: "", conf: 0))
            return
        }

        // Debug save
        if let out = debugOut {
            let outURL = out.appendingPathComponent(String(format: "stack_%06d.png", centerFrame))
            try? savePNG(finalCG, to: outURL)
        }

        // OCR
        do {
            let r = try ocrCGImage(finalCG, languages: languages)
            let clean = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let ok = (clean.count >= minChars) && (r.bestConfidence >= minConf)
            results.append(FrameOCR(frame: centerFrame, ok: ok, text: clean, conf: r.bestConfidence))
        } catch {
            results.append(FrameOCR(frame: centerFrame, ok: false, text: "", conf: 0))
        }
    }
}

results.sort { $0.frame < $1.frame }

// MARK: - Build intervals + choose best text per interval (highest confidence)

struct Interval {
    let startFrame: Int
    let endFrame: Int
    let bestText: String
    let bestConf: Float
    let bestFrame: Int
}

var intervals: [Interval] = []
var currentStart: Int? = nil
var currentBestText: String = ""
var currentBestConf: Float = -1
var currentBestFrame: Int = -1

var prevFrame: Int? = nil
let maxGap = step // if scanning with step>1, treat <=step as contiguous

for r in results {
    if r.ok {
        if currentStart == nil {
            currentStart = r.frame
            currentBestText = r.text
            currentBestConf = r.conf
            currentBestFrame = r.frame
        } else if let pf = prevFrame, r.frame > pf + maxGap {
            // close previous interval
            intervals.append(Interval(
                startFrame: currentStart!,
                endFrame: pf,
                bestText: currentBestText,
                bestConf: currentBestConf,
                bestFrame: currentBestFrame
            ))
            // start new
            currentStart = r.frame
            currentBestText = r.text
            currentBestConf = r.conf
            currentBestFrame = r.frame
        } else {
            // still within interval; update best text
            if r.conf > currentBestConf && !r.text.isEmpty {
                currentBestConf = r.conf
                currentBestText = r.text
                currentBestFrame = r.frame
            }
        }
    } else {
        if let s = currentStart, let pf = prevFrame {
            intervals.append(Interval(
                startFrame: s,
                endFrame: pf,
                bestText: currentBestText,
                bestConf: currentBestConf,
                bestFrame: currentBestFrame
            ))
            currentStart = nil
            currentBestText = ""
            currentBestConf = -1
            currentBestFrame = -1
        }
    }
    prevFrame = r.frame
}

// close tail
if let s = currentStart, let pf = prevFrame {
    intervals.append(Interval(
        startFrame: s,
        endFrame: pf,
        bestText: currentBestText,
        bestConf: currentBestConf,
        bestFrame: currentBestFrame
    ))
}

print("FPS=\(fps) (default 30 if omitted), stack=\(stackCount), step=\(step), maxFrame=\(maxFrame), oneBased=\(oneBased)")
if step != 1 {
    print("Note: step=\(step) means interval boundaries are approximate. Use --step 1 for best timing.")
}
print("Detected text intervals:")

if intervals.isEmpty {
    print("- none")
    exit(0)
}

for iv in intervals {
    let t0 = frameToSecondsStart(iv.startFrame, fps: fps, oneBased: oneBased)
    let t1 = frameToSecondsEndExclusive(iv.endFrame, fps: fps, oneBased: oneBased)

    let textOneLine = collapseWhitespace(iv.bestText)
    // Print: time interval + text
    print(String(format: "- %.3fs → %.3fs  (frames %d–%d, best@%d conf=%.2f)",
                 t0, t1, iv.startFrame, iv.endFrame, iv.bestFrame, iv.bestConf))
    print("  \(textOneLine)")
}

