// AdvancedResample.swift
// Build: swiftc AdvancedResample.swift -o advancedResample
// Run:   ./advancedResample input=<file-or-folder> output=<folder> scale=<multiplier> [steps=...] [flatten=...] [core=...] [binarize=...] [thresh=...] [shrink=...] [thin=...] [regrow=...] > results.json

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import Metal
import Vision

// -------------------- Utilities --------------------

func usage() {
    print("""
Usage:
  advancedResample input=<file-or-folder> output=<folder> scale=<multiplier> [steps=...] [flatten=...] [core=...] [binarize=...] [thresh=...] [shrink=...] [thin=...] [regrow=...]

Required:
  input   = path to an image file OR a folder of images
  output  = destination folder for final prepped images
  scale   = multiplier (e.g. 2, 1.5, 10, 12)

Optional:
  steps   = false | true | <folder>
            - true => saves steps to output/steps
            - <folder> => saves steps there

  flatten = true|false (default true)
            - illumination flattening (helps glare/shading on originals)

  core    = pixels (>=0, default 0)
            - GRAYSCALE core-tighten BEFORE threshold.
              Removes halo/spread by brightening edges while keeping darkest cores.

  binarize = true|false (default true)
            - false => skip thresholding; feed grayscale to Vision OCR (keeps nuance)

  thresh  = 0..1 (default: Otsu if binarize=true)
            - manual threshold override (often helpful with glare)

  shrink  = pixels (>=0, default 0) [only if binarize=true]
            - shrink black ink AFTER threshold (removes remaining fat)

  thin    = false | true | <maxIters> (default false) [only if binarize=true]
            - Zhang–Suen thinning (skeletonization)

  regrow  = pixels (>=0, default 0) [only if binarize=true]
            - regrow black slightly after thinning/shrink

Examples:
  ./advancedResample input=./images output=./out scale=12 flatten=true core=2 binarize=false steps=true > results.json
  ./advancedResample input=./images output=./out scale=12 flatten=true core=2 binarize=true thin=true regrow=2 steps=true > results.json
""")
}

func ensureDirectory(_ url: URL) throws {
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) {
        if !isDir.boolValue {
            throw NSError(domain: "advancedResample", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Folder path exists but is not a folder: \(url.path)"
            ])
        }
    } else {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
}

func isImageFile(_ url: URL) -> Bool {
    let exts = ["jpg","jpeg","png","heic","heif","tif","tiff","bmp","gif","webp"]
    return exts.contains(url.pathExtension.lowercased())
}

// Finder-like Name ascending sort (natural / localized)
func listImageFiles(in folder: URL) -> [URL] {
    guard let items = try? FileManager.default.contentsOfDirectory(
        at: folder,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else { return [] }

    return items
        .filter { !$0.hasDirectoryPath && isImageFile($0) }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
}

func scaleTag(_ f: Float) -> String {
    let s = String(format: "%.2f", f)
    let trimmed = s
        .replacingOccurrences(of: #"(\.00)$"#, with: "", options: .regularExpression)
        .replacingOccurrences(of: #"(\.\d)0$"#, with: "$1", options: .regularExpression)
    return "\(trimmed)x"
}

func parseBoolish(_ s: String) -> Bool? {
    let v = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if ["1","true","yes","y","on"].contains(v) { return true }
    if ["0","false","no","n","off"].contains(v) { return false }
    return nil
}

// -------------------- Image I/O --------------------

struct LoadedImage {
    let ciImage: CIImage
    let inputURL: URL
    let name: String
}

func loadCIImage(url: URL) -> LoadedImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }

    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let exifOrientation = props?[kCGImagePropertyOrientation] as? UInt32 ?? 1

    guard var ci = CIImage(contentsOf: url) else { return nil }
    ci = ci.oriented(forExifOrientation: Int32(exifOrientation))

    return LoadedImage(ciImage: ci, inputURL: url, name: url.lastPathComponent)
}

func writePNG(cgImage: CGImage, to url: URL) -> Bool {
    let uti = UTType.png.identifier
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, uti as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(dest, cgImage, nil)
    return CGImageDestinationFinalize(dest)
}

// -------------------- OCR (Vision) --------------------

func recognizeText(cgImage: CGImage) -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
        guard let results = request.results, !results.isEmpty else { return "" }
        let sorted = results.sorted { a, b in a.boundingBox.minY > b.boundingBox.minY }
        let lines = sorted.compactMap { $0.topCandidates(1).first?.string }
        return lines.joined(separator: "\n")
    } catch {
        return ""
    }
}

// -------------------- Thinning (Zhang–Suen) --------------------
// Foreground = 1 (black ink), Background = 0 (white)

func zhangSuenThin(_ src: [UInt8], width: Int, height: Int, maxIterations: Int) -> [UInt8] {
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

func cgImageToGrayBytes(_ cg: CGImage) -> (bytes: [UInt8], width: Int, height: Int)? {
    let w = cg.width
    let h = cg.height
    var buf = [UInt8](repeating: 0, count: w * h)

    guard let ctx = CGContext(
        data: &buf,
        width: w,
        height: h,
        bitsPerComponent: 8,
        bytesPerRow: w,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else { return nil }

    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (buf, w, h)
}

func grayBytesToCGImage(_ bytes: [UInt8], width: Int, height: Int) -> CGImage? {
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

// -------------------- Preprocessor --------------------

final class OCRPreprocessor {
    private let context: CIContext

    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            context = CIContext(mtlDevice: device)
        } else {
            context = CIContext()
        }
    }

    func render(_ image: CIImage) -> CGImage? {
        context.createCGImage(image, from: image.extent.integral)
    }

    private func meanBrightness(_ image: CIImage) -> Double {
        let avg = CIFilter.areaAverage()
        avg.inputImage = image
        avg.extent = image.extent

        guard let out = avg.outputImage,
              let cg = context.createCGImage(out, from: CGRect(x: 0, y: 0, width: 1, height: 1)) else { return 0.0 }

        var px: [UInt8] = [0, 0, 0, 0]
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        guard let ctx = CGContext(
            data: &px,
            width: 1, height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: cs,
            bitmapInfo: bitmapInfo
        ) else { return 0.0 }

        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        let b = Double(px[0]), g = Double(px[1]), r = Double(px[2])
        return (r + g + b) / (3.0 * 255.0)
    }

    func preprocessForOCR(_ input: CIImage,
                          scale: Float,
                          flatten: Bool,
                          corePixels: Float,
                          binarize: Bool,
                          manualThresh: Float?,     // 0..1; nil => Otsu
                          shrinkPixels: Float,
                          thinEnabled: Bool,
                          thinMaxIters: Int,        // 0 => until stable
                          regrowPixels: Float,
                          steps: inout [String: CIImage]?) -> CIImage {

        let extent = input.extent

        // 1) grayscale
        var img = input.applyingFilter("CIColorControls", parameters: [
            kCIInputSaturationKey: 0.0,
            kCIInputContrastKey: 1.0,
            kCIInputBrightnessKey: 0.0
        ]).cropped(to: extent)
        steps?["01_grayscale"] = img

        // 2) optional illumination flattening
        if flatten {
            let clamped = img.clampedToExtent()
            let blur = CIFilter.gaussianBlur()
            blur.inputImage = clamped
            let minDim = min(extent.width, extent.height)
            blur.radius = Float(max(10.0, Double(minDim * 0.03)))
            let blurred = (blur.outputImage ?? clamped).cropped(to: extent)

            img = img.applyingFilter("CIDivideBlendMode", parameters: [kCIInputBackgroundImageKey: blurred])
            img = img.applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 0.0,
                kCIInputContrastKey: 1.4,
                kCIInputBrightnessKey: 0.0
            ]).cropped(to: img.extent.integral)
        }
        steps?["02_flattened"] = img

        // 3) upscale FIRST
        let lz = CIFilter.lanczosScaleTransform()
        lz.inputImage = img
        lz.scale = scale
        lz.aspectRatio = 1.0
        img = (lz.outputImage ?? img).cropped(to: (lz.outputImage ?? img).extent.integral)
        steps?["03_upscaled"] = img

        // 3b) grayscale core-tighten BEFORE threshold
        if corePixels >= 1 {
            let core = CIFilter.morphologyRectangleMaximum() // max filter brightens edges -> suppresses dark halo
            core.inputImage = img
            core.width = corePixels
            core.height = corePixels
            img = (core.outputImage ?? img).cropped(to: (core.outputImage ?? img).extent.integral)
        }
        steps?["03b_core"] = img

        // If not binarizing, return grayscale for OCR
        if !binarize {
            return img.cropped(to: img.extent.integral)
        }

        // 4) threshold AFTER upscale (and after core-tighten)
        if let t = manualThresh {
            img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": max(0, min(1, t))])
                .cropped(to: img.extent.integral)
        } else if #available(macOS 11.0, *) {
            let otsu = CIFilter.colorThresholdOtsu()
            otsu.inputImage = img
            img = (otsu.outputImage ?? img).cropped(to: (otsu.outputImage ?? img).extent.integral)
        } else {
            img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": 0.5])
                .cropped(to: img.extent.integral)
        }

        // normalize to black-on-white
        if meanBrightness(img) < 0.5 {
            img = img.applyingFilter("CIColorInvert").cropped(to: img.extent.integral)
        }
        steps?["04_threshold"] = img

        // 5) optional shrink black (remove remaining fat)
        if shrinkPixels >= 1 {
            let shrink = CIFilter.morphologyRectangleMaximum() // white expands => black contracts
            shrink.inputImage = img
            shrink.width = shrinkPixels
            shrink.height = shrinkPixels
            img = (shrink.outputImage ?? img).cropped(to: (shrink.outputImage ?? img).extent.integral)
        }
        steps?["05_shrunk"] = img

        // 6) thinning
        if thinEnabled, let cg = render(img), let g = cgImageToGrayBytes(cg) {
            var bin = [UInt8](repeating: 0, count: g.bytes.count)
            for i in 0..<g.bytes.count { bin[i] = (g.bytes[i] < 128) ? 1 : 0 }

            let thinned = zhangSuenThin(bin, width: g.width, height: g.height, maxIterations: thinMaxIters)

            var out = [UInt8](repeating: 255, count: thinned.count)
            for i in 0..<thinned.count { out[i] = (thinned[i] == 1) ? 0 : 255 }

            if let cg2 = grayBytesToCGImage(out, width: g.width, height: g.height) {
                img = CIImage(cgImage: cg2).cropped(to: CGRect(x: 0, y: 0, width: g.width, height: g.height))
            }
        }
        steps?["06_thinned"] = img

        // 7) optional regrow
        if regrowPixels >= 1 {
            let grow = CIFilter.morphologyRectangleMinimum() // black expands
            grow.inputImage = img
            grow.width = regrowPixels
            grow.height = regrowPixels
            img = (grow.outputImage ?? img).cropped(to: (grow.outputImage ?? img).extent.integral)
        }
        steps?["07_regrown"] = img

        return img.cropped(to: img.extent.integral)
    }
}

// -------------------- JSON output --------------------

struct OCRResult: Codable {
    let name: String
    let input_path: String
    let output_path: String
    let ocr: String
    let steps_dir: String?
    let steps: [String: String]?
}

// -------------------- Main (name=value args) --------------------

let args = CommandLine.arguments
guard args.count >= 2 else { usage(); exit(1) }

var kv: [String: String] = [:]
for a in args.dropFirst() {
    guard let eq = a.firstIndex(of: "=") else {
        fputs("Error: all arguments must be name=value (got: \(a))\n", stderr)
        usage()
        exit(1)
    }
    let key = String(a[..<eq]).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let val = String(a[a.index(after: eq)...]).trimmingCharacters(in: .whitespacesAndNewlines)
    kv[key] = val
}

guard let inputStr = kv["input"], !inputStr.isEmpty,
      let outputStr = kv["output"], !outputStr.isEmpty,
      let scaleStr = kv["scale"], let scale = Float(scaleStr), scale > 0 else {
    fputs("Error: missing required input=, output=, or valid scale=\n", stderr)
    usage()
    exit(1)
}

let inputURL = URL(fileURLWithPath: inputStr)
let outputDir = URL(fileURLWithPath: outputStr)

// steps
let stepsVal = kv["steps"]
let stepsEnabled: Bool
let stepsDir: URL?
if let s = stepsVal {
    if let b = parseBoolish(s) {
        stepsEnabled = b
        stepsDir = b ? outputDir.appendingPathComponent("steps") : nil
    } else {
        stepsEnabled = true
        stepsDir = URL(fileURLWithPath: s)
    }
} else {
    stepsEnabled = false
    stepsDir = nil
}

let flatten: Bool = {
    if let s = kv["flatten"], let b = parseBoolish(s) { return b }
    return true
}()

let corePixels: Float = {
    if let s = kv["core"], let v = Float(s), v >= 0 { return v }
    return 0
}()

let binarize: Bool = {
    if let s = kv["binarize"], let b = parseBoolish(s) { return b }
    return true
}()

let manualThresh: Float? = {
    if let s = kv["thresh"], let v = Float(s) { return v }
    return nil
}()

let shrinkPixels: Float = {
    if let s = kv["shrink"], let v = Float(s), v >= 0 { return v }
    return 0
}()

let regrowPixels: Float = {
    if let s = kv["regrow"], let v = Float(s), v >= 0 { return v }
    return 0
}()

var thinEnabled = false
var thinMaxIters = 0
if let s = kv["thin"] {
    if let b = parseBoolish(s) {
        thinEnabled = b
    } else if let n = Int(s) {
        thinEnabled = n != 0
        thinMaxIters = max(0, n)
    }
}

do { try ensureDirectory(outputDir) }
catch { fputs("Error: \(error.localizedDescription)\n", stderr); exit(2) }

if stepsEnabled, let sd = stepsDir {
    do { try ensureDirectory(sd) }
    catch { fputs("Error: \(error.localizedDescription)\n", stderr); exit(2) }
}

var isDir: ObjCBool = false
guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDir) else {
    fputs("Error: input not found: \(inputURL.path)\n", stderr)
    exit(3)
}

let files: [URL] = isDir.boolValue ? listImageFiles(in: inputURL) : [inputURL]
if files.isEmpty {
    fputs("No images found at: \(inputURL.path)\n", stderr)
    exit(0)
}

let pre = OCRPreprocessor()
let tag = scaleTag(scale)

func safeFolderName(base: String, ext: String) -> String {
    let e = ext.isEmpty ? "img" : ext
    return "\(base)_\(e)".replacingOccurrences(of: "/", with: "_")
}

var results: [OCRResult] = []
results.reserveCapacity(files.count)

for file in files {
    autoreleasepool {
        guard let loaded = loadCIImage(url: file) else {
            fputs("Skip (can't load): \(file.lastPathComponent)\n", stderr)
            return
        }

        var stepImages: [String: CIImage]? = stepsEnabled ? [:] : nil

        let processed = pre.preprocessForOCR(
            loaded.ciImage,
            scale: scale,
            flatten: flatten,
            corePixels: corePixels,
            binarize: binarize,
            manualThresh: manualThresh,
            shrinkPixels: shrinkPixels,
            thinEnabled: thinEnabled,
            thinMaxIters: thinMaxIters,
            regrowPixels: regrowPixels,
            steps: &stepImages
        )

        guard let cg = pre.render(processed) else {
            fputs("Skip (can't render): \(file.lastPathComponent)\n", stderr)
            return
        }

        let base = file.deletingPathExtension().lastPathComponent
        let outName = "\(base)_prep_\(tag).png"
        let outURL = outputDir.appendingPathComponent(outName)
        _ = writePNG(cgImage: cg, to: outURL)

        let text = recognizeText(cgImage: cg)

        var stepsOutDirPath: String? = nil
        var stepPaths: [String: String]? = nil

        if stepsEnabled, let sd = stepsDir, let imgs = stepImages {
            let subdir = sd.appendingPathComponent(safeFolderName(base: base, ext: file.pathExtension.lowercased()))
            try? ensureDirectory(subdir)
            stepsOutDirPath = subdir.path

            stepPaths = [:]
            for key in imgs.keys.sorted() {
                guard let ci = imgs[key], let stepCG = pre.render(ci) else { continue }
                let stepURL = subdir.appendingPathComponent("\(key).png")
                if writePNG(cgImage: stepCG, to: stepURL) {
                    stepPaths?[key] = stepURL.path
                }
            }
        }

        results.append(OCRResult(
            name: loaded.name,
            input_path: loaded.inputURL.path,
            output_path: outURL.path,
            ocr: text,
            steps_dir: stepsOutDirPath,
            steps: stepPaths
        ))
    }
}

let enc = JSONEncoder()
enc.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
do {
    let data = try enc.encode(results)
    print(String(data: data, encoding: .utf8) ?? "[]")
} catch {
    fputs("Failed to encode JSON: \(error)\n", stderr)
    exit(5)
}

