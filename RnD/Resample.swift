// Resample.swift
// Build: swiftc Resample.swift -o resample
// Run:   ./resample /path/in /path/out 2
//        ./resample /path/in /path/out 1.5
//        ./resample /path/in /path/out 2x

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import Metal

// ---------- Core processing (portable to iOS) ----------

final class DenoiseUpscaler {
    private let context: CIContext

    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            context = CIContext(mtlDevice: device)
        } else {
            context = CIContext()
        }
    }

    /// Denoise -> Lanczos scale by factor (no crop).
    func process(_ input: CIImage,
                 scaleFactor: Float,
                 noiseLevel: Float = 0.03,
                 sharpness: Float = 0.40) -> CIImage {

        // 1) Fast denoise
        let nr = CIFilter.noiseReduction()
        nr.inputImage = input
        nr.noiseLevel = noiseLevel
        nr.sharpness = sharpness
        var out = nr.outputImage ?? input

        // 2) Lanczos scale by multiplier
        let lz = CIFilter.lanczosScaleTransform()
        lz.inputImage = out
        lz.scale = scaleFactor
        lz.aspectRatio = 1.0
        out = lz.outputImage ?? out

        // Ensure pixel-aligned extent for clean renders
        out = out.cropped(to: out.extent.integral)
        return out
    }

    func render(_ image: CIImage) -> CGImage? {
        let rect = image.extent.integral
        return context.createCGImage(image, from: rect)
    }
}

// ---------- Image IO helpers ----------

struct LoadedImage {
    let ciImage: CIImage
    let uti: String?
}

func loadImage(url: URL) -> LoadedImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }

    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let exifOrientation = props?[kCGImagePropertyOrientation] as? UInt32 ?? 1

    guard var ci = CIImage(contentsOf: url) else { return nil }
    ci = ci.oriented(forExifOrientation: Int32(exifOrientation))

    let uti = CGImageSourceGetType(src) as String?
    return LoadedImage(ciImage: ci, uti: uti)
}

func writeImage(cgImage: CGImage, to url: URL, preferredUTI: String?) -> Bool {
    let fallback = UTType.jpeg.identifier
    let outUTI = preferredUTI ?? fallback

    func finalize(_ dest: CGImageDestination) -> Bool {
        let options = [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary
        CGImageDestinationAddImage(dest, cgImage, options)
        return CGImageDestinationFinalize(dest)
    }

    if let dest = CGImageDestinationCreateWithURL(url as CFURL, outUTI as CFString, 1, nil) {
        return finalize(dest)
    } else if let dest2 = CGImageDestinationCreateWithURL(url as CFURL, fallback as CFString, 1, nil) {
        return finalize(dest2)
    }
    return false
}

func isImageFile(_ url: URL) -> Bool {
    let exts = ["jpg","jpeg","png","heic","heif","tif","tiff","bmp","gif","webp"]
    return exts.contains(url.pathExtension.lowercased())
}

func ensureDirectory(_ url: URL) throws {
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) {
        if !isDir.boolValue {
            throw NSError(domain: "resample", code: 2, userInfo: [NSLocalizedDescriptionKey: "Output exists but is not a folder: \(url.path)"])
        }
    } else {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
}

func listImageFiles(in folder: URL) -> [URL] {
    guard let items = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
        return []
    }
    return items.filter { !$0.hasDirectoryPath && isImageFile($0) }
}

// ---------- CLI ----------

func usage() {
    print("""
Usage:
  resample <input_file_or_folder> <output_folder> <scale>

Scale can be decimal and may optionally end with 'x'.
Examples:
  ./resample ./photo.jpg ./out 2
  ./resample ./images   ./out 1.5
  ./resample ./images   ./out 2x
""")
}

func parseScale(_ s: String) -> Float? {
    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
    let noX = trimmed.hasSuffix("x") || trimmed.hasSuffix("X") ? String(trimmed.dropLast()) : trimmed
    return Float(noX)
}

let args = CommandLine.arguments
guard args.count == 4 else { usage(); exit(1) }

let inputURL = URL(fileURLWithPath: args[1])
let outputDir = URL(fileURLWithPath: args[2])

guard let scale = parseScale(args[3]), scale > 0 else {
    fputs("Error: scale must be > 0 (e.g. 2, 1.5, 2x)\n", stderr)
    exit(1)
}

do { try ensureDirectory(outputDir) }
catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(2)
}

var isDir: ObjCBool = false
let exists = FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDir)
guard exists else {
    fputs("Error: input not found: \(inputURL.path)\n", stderr)
    exit(3)
}

let files: [URL] = isDir.boolValue ? listImageFiles(in: inputURL) : [inputURL]
if files.isEmpty {
    fputs("No images found at: \(inputURL.path)\n", stderr)
    exit(0)
}

let upscaler = DenoiseUpscaler()

// For filenames: 2 -> "2x", 1.5 -> "1.5x", 2.25 -> "2.25x"
func scaleTag(_ f: Float) -> String {
    let s = String(format: "%.2f", f)
    let trimmed = s.replacingOccurrences(of: #"(\.00)$"#, with: "", options: .regularExpression)
                    .replacingOccurrences(of: #"(\.\d)0$"#, with: "$1", options: .regularExpression)
    return "\(trimmed)x"
}

let tag = scaleTag(scale)

for file in files {
    guard let loaded = loadImage(url: file) else {
        fputs("Skip (can't load): \(file.lastPathComponent)\n", stderr)
        continue
    }

    let outCI = upscaler.process(loaded.ciImage, scaleFactor: scale)
    guard let cg = upscaler.render(outCI) else {
        fputs("Skip (can't render): \(file.lastPathComponent)\n", stderr)
        continue
    }

    let base = file.deletingPathExtension().lastPathComponent
    let ext = file.pathExtension.isEmpty ? "jpg" : file.pathExtension
    let outName = "\(base)_\(tag).\(ext)"
    let outURL = outputDir.appendingPathComponent(outName)

    if writeImage(cgImage: cg, to: outURL, preferredUTI: loaded.uti) {
        print("Wrote: \(outURL.path)")
    } else {
        fputs("Failed to write: \(outURL.path)\n", stderr)
    }
}

