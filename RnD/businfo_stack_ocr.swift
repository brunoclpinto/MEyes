import Foundation
import Vision
import CoreImage
import ImageIO
import CoreGraphics

// MARK: - Utils

struct FrameFile {
    let frame: Int
    let url: URL
}

func parseFrameNumber(filename: String) -> Int? {
    // BusInfo000123.jpg (exactly 6 digits)
    let pattern = #"^BusInfo(\d{6})\.jpg$"#
    let re = try? NSRegularExpression(pattern: pattern)
    let range = NSRange(filename.startIndex..<filename.endIndex, in: filename)
    guard
        let m = re?.firstMatch(in: filename, options: [], range: range),
        m.numberOfRanges == 2,
        let r = Range(m.range(at: 1), in: filename)
    else { return nil }
    return Int(filename[r])
}

func usage() -> Never {
    print("""
    Usage:
      businfo_stack_ocr <folder>
        [--fps 30] [--maxframe 200] [--onebased]
        [--lang pt-PT,en-US]
        [--stack 21] [--step 1]
        [--minchars 3] [--minconf 0.50]
        [--sharpen 0.6] [--radius 2.0] [--contrast 1.15]
        [--debugout <folder>]

    Notes:
      - --stack is number of frames to stack (odd recommended). Default 21 (~0.7s at 30fps).
      - Alignment uses VNTranslationalImageRegistrationRequest (translation-only).
      - If scale/perspective changes are strong, you may need homography (next step).

    Examples:
      businfo_stack_ocr .                 (run inside the frames folder)
      businfo_stack_ocr . --stack 31
      businfo_stack_ocr . --step 2
      businfo_stack_ocr . --debugout ./dbg
    """)
    exit(2)
}

func ensureDir(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
}

func savePNG(_ cgImage: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, kUTTypePNG, 1, nil) else {
        throw NSError(domain: "save", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create image destination"])
    }
    CGImageDestinationAddImage(dest, cgImage, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "save", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot finalize image destination"])
    }
}

// MARK: - Pad to canvas (handles different crop sizes)

func paddedToCanvas(_ img: CIImage, canvasRect: CGRect) -> (padded: CIImage, mask: CIImage) {
    let extent = img.extent.integral
    let bg = CIImage(color: .black).cropped(to: canvasRect)

    let x = (canvasRect.width - extent.width) * 0.5
    let y = (canvasRect.height - extent.height) * 0.5

    // move image so it is centered on the canvas
    let t = CGAffineTransform(translationX: x - extent.origin.x, y: y - extent.origin.y)
    let moved = img.transformed(by: t)

    // Mask: white where original pixels exist, black elsewhere
    let whiteRect = CIImage(color: .white).cropped(to: CGRect(x: x, y: y, width: extent.width, height: extent.height))
    let mask = whiteRect.composited(over: bg).cropped(to: canvasRect)

    let padded = moved.composited(over: bg).cropped(to: canvasRect)
    return (padded, mask)
}

// MARK: - Render CIImage -> grayscale buffer (R8)

final class GrayRenderer {
    let context: CIContext
    let grayCS = CGColorSpaceCreateDeviceGray()

    init() {
        self.context = CIContext(options: [
            .cacheIntermediates: false
        ])
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

// MARK: - Alignment (translation-only) using Vision

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
        guard
            let obs = request.results?.first as? VNImageTranslationAlignmentObservation
        else { return nil }

        let t = obs.alignmentTransform
        let aligned = moving.transformed(by: t).cropped(to: canvasRect)
        let alignedMask = movingMask.transformed(by: t).cropped(to: canvasRect)
        return (aligned, alignedMask)
    } catch {
        return nil
    }
}

// MARK: - Median stack (with mask)

func medianStack(buffers: [[UInt8]], masks: [[UInt8]], width: Int, height: Int) -> [UInt8] {
    let n = buffers.count
    let total = width * height
    var out = [UInt8](repeating: 0, count: total)
    var tmp = [UInt8]()
    tmp.reserveCapacity(n)

    for i in 0..<total {
        tmp.removeAll(keepingCapacity: true)
        for k in 0..<n {
            if masks[k][i] > 0 {
                tmp.append(buffers[k][i])
            }
        }
        if tmp.isEmpty {
            out[i] = 0
        } else {
            tmp.sort()
            out[i] = tmp[tmp.count / 2]
        }
    }
    return out
}

// MARK: - OCR

struct OCRResult {
    let hasText: Bool
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
    return OCRResult(hasText: !text.isEmpty, text: text, bestConfidence: bestConf)
}

// MARK: - Main

let args = Array(CommandLine.arguments.dropFirst())
guard !args.isEmpty else { usage() }

var folderPath: String?
var fps: Double = 30.0
var maxFrame: Int = 200
var oneBased = false
var languages = ["en-US", "pt-PT"]

var stackCount = 21            // frames to stack (odd recommended)
var step = 1                   // process every Nth frame

var minChars = 3
var minConf: Float = 0.50

var sharpen: Double = 0.6
var radius: Double = 2.0
var contrast: Double = 1.15

var debugOut: URL? = nil

var i = 0
while i < args.count {
    let a = args[i]
    switch a {
    case "--fps":
        guard i+1 < args.count, let v = Double(args[i+1]) else { usage() }
        fps = v; i += 2
    case "--maxframe":
        guard i+1 < args.count, let v = Int(args[i+1]) else { usage() }
        maxFrame = v; i += 2
    case "--onebased":
        oneBased = true; i += 1
    case "--lang":
        guard i+1 < args.count else { usage() }
        languages = args[i+1].split(separator: ",").map { String($0) }
        i += 2
    case "--stack":
        guard i+1 < args.count, let v = Int(args[i+1]), v >= 3 else { usage() }
        stackCount = v; i += 2
    case "--step":
        guard i+1 < args.count, let v = Int(args[i+1]), v >= 1 else { usage() }
        step = v; i += 2
    case "--minchars":
        guard i+1 < args.count, let v = Int(args[i+1]), v >= 0 else { usage() }
        minChars = v; i += 2
    case "--minconf":
        guard i+1 < args.count, let v = Float(args[i+1]) else { usage() }
        minConf = v; i += 2
    case "--sharpen":
        guard i+1 < args.count, let v = Double(args[i+1]) else { usage() }
        sharpen = v; i += 2
    case "--radius":
        guard i+1 < args.count, let v = Double(args[i+1]) else { usage() }
        radius = v; i += 2
    case "--contrast":
        guard i+1 < args.count, let v = Double(args[i+1]) else { usage() }
        contrast = v; i += 2
    case "--debugout":
        guard i+1 < args.count else { usage() }
        debugOut = URL(fileURLWithPath: args[i+1])
        i += 2
    default:
        if folderPath == nil { folderPath = a; i += 1 }
        else { usage() }
    }
}

guard let folder = folderPath else { usage() }

let fm = FileManager.default
let dirURL = URL(fileURLWithPath: folder)

let files = try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])

var frames: [FrameFile] = files.compactMap { url in
    let name = url.lastPathComponent
    guard let f = parseFrameNumber(filename: name), f <= maxFrame else { return nil }
    return FrameFile(frame: f, url: url)
}.sorted { $0.frame < $1.frame }

if frames.isEmpty {
    fputs("No BusInfo######.jpg files (<= \(maxFrame)) found in: \(dirURL.path)\n", stderr)
    exit(1)
}

if let out = debugOut {
    try ensureDir(out)
}

let renderer = GrayRenderer()
let ciContext = renderer.context

// Load as CIImages, track max canvas size
var ciFrames: [(frame: Int, image: CIImage)] = []
ciFrames.reserveCapacity(frames.count)

var maxW: Int = 0
var maxH: Int = 0

for f in frames {
    if let img = CIImage(contentsOf: f.url) {
        let e = img.extent.integral
        maxW = max(maxW, Int(e.width))
        maxH = max(maxH, Int(e.height))
        ciFrames.append((f.frame, img))
    }
}

if ciFrames.isEmpty {
    fputs("Could not load any images as CIImage.\n", stderr)
    exit(1)
}

let canvasRect = CGRect(x: 0, y: 0, width: maxW, height: maxH)

func frameToSecondsStart(_ frame: Int) -> Double {
    let f = oneBased ? (frame - 1) : frame
    return Double(max(f, 0)) / fps
}
func frameToSecondsEndExclusive(_ frameInclusive: Int) -> Double {
    // cover frame duration: (last+1)/fps (0-based)
    let f = oneBased ? frameInclusive : (frameInclusive + 1)
    return Double(max(f, 0)) / fps
}

struct Interval { let startFrame: Int; let endFrame: Int }

var recognized: [(frame: Int, ok: Bool)] = []

// Process frames with stride
let half = max(1, stackCount / 2)

for idx in stride(from: 0, to: ciFrames.count, by: step) {
    autoreleasepool {
        let (centerFrame, centerImg) = ciFrames[idx]
        let (refPadded, refMask) = paddedToCanvas(centerImg, canvasRect: canvasRect)

        // Collect aligned buffers
        var imgBuffers: [[UInt8]] = []
        var maskBuffers: [[UInt8]] = []
        imgBuffers.reserveCapacity(stackCount)
        maskBuffers.reserveCapacity(stackCount)

        // Always include reference first
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
                    let b = renderer.renderR8(aligned, rect: canvasRect, width: maxW, height: maxH)
                    let m = renderer.renderR8(alignedMask, rect: canvasRect, width: maxW, height: maxH)
                    imgBuffers.append(b)
                    maskBuffers.append(m)
                }
            }
        }

        // If we couldn't align enough frames, still try OCR on reference
        let stackedBuf: [UInt8]
        if imgBuffers.count >= 3 {
            stackedBuf = medianStack(buffers: imgBuffers, masks: maskBuffers, width: maxW, height: maxH)
        } else {
            stackedBuf = refBuf
        }

        guard let stackedCG = renderer.makeGrayCGImage(from: stackedBuf, width: maxW, height: maxH) else {
            recognized.append((frame: centerFrame, ok: false))
            return
        }

        // Optional enhancement: unsharp + contrast
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
            recognized.append((frame: centerFrame, ok: false))
            return
        }

        // Debug save (optional)
        if let out = debugOut {
            let outURL = out.appendingPathComponent(String(format: "stack_%06d.png", centerFrame))
            try? savePNG(finalCG, to: outURL)
        }

        // OCR
        do {
            let r = try ocrCGImage(finalCG, languages: languages)
            let ok = (r.text.count >= minChars) && (r.bestConfidence >= minConf)
            recognized.append((frame: centerFrame, ok: ok))
        } catch {
            recognized.append((frame: centerFrame, ok: false))
        }
    }
}

// Build intervals (contiguous within expected gaps)
recognized.sort { $0.frame < $1.frame }

var intervals: [Interval] = []
var currentStart: Int? = nil
var prevFrame: Int? = nil
let maxGap = step  // treat gaps up to 'step' as contiguous when scanning with stride

for (frame, ok) in recognized {
    if ok {
        if currentStart == nil {
            currentStart = frame
        } else if let pf = prevFrame, frame > pf + maxGap {
            intervals.append(Interval(startFrame: currentStart!, endFrame: pf))
            currentStart = frame
        }
    } else {
        if let s = currentStart, let pf = prevFrame {
            intervals.append(Interval(startFrame: s, endFrame: pf))
            currentStart = nil
        }
    }
    prevFrame = frame
}
if let s = currentStart, let pf = prevFrame {
    intervals.append(Interval(startFrame: s, endFrame: pf))
}

print("Recognized intervals (stack=\(stackCount), step=\(step), fps=\(fps))")
if intervals.isEmpty {
    print("- none")
} else {
    var total: Double = 0
    for iv in intervals {
        let t0 = frameToSecondsStart(iv.startFrame)
        let t1 = frameToSecondsEndExclusive(iv.endFrame)
        total += max(0, t1 - t0)
        print(String(format: "- %.3fs → %.3fs  (frames %d–%d)", t0, t1, iv.startFrame, iv.endFrame))
    }
    print(String(format: "Total recognized duration: %.3fs", total))
}

