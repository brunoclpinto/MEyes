import Foundation
import Vision
import ImageIO

func loadCGImage(from url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func ocrImage(at url: URL, languages: [String]) throws -> [String] {
    guard let cgImage = loadCGImage(from: url) else {
        throw NSError(domain: "ocr", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load image: \(url.path)"])
    }

    var lines: [String] = []

    let request = VNRecognizeTextRequest { req, err in
        if let err = err {
            fputs("OCR error for \(url.lastPathComponent): \(err)\n", stderr)
            return
        }
        guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
        for obs in observations {
            if let best = obs.topCandidates(1).first {
                lines.append(best.string)
            }
        }
    }

    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = languages

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])

    return lines
}

func usage() -> Never {
    let msg = """
    Usage:
      ocr <folder_path> [--lang pt-PT,en-US] [--json]

    Examples:
      ocr ./images
      ocr ./images --lang pt-PT,en-US
      ocr ./images --json
    """
    print(msg)
    exit(2)
}

let args = Array(CommandLine.arguments.dropFirst())

guard !args.isEmpty else { usage() }

var folderPath: String?
var languages = ["en-US", "pt-PT"]
var jsonOutput = false

var i = 0
while i < args.count {
    let a = args[i]
    if a == "--lang" {
        guard i + 1 < args.count else { usage() }
        languages = args[i + 1].split(separator: ",").map { String($0) }
        i += 2
        continue
    } else if a == "--json" {
        jsonOutput = true
        i += 1
        continue
    } else if folderPath == nil {
        folderPath = a
        i += 1
        continue
    } else {
        usage()
    }
}

guard let folder = folderPath else { usage() }

let fm = FileManager.default
let dirURL = URL(fileURLWithPath: folder)

let files = try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    .filter { url in
        let ext = url.pathExtension.lowercased()
        return ext == "jpg" || ext == "jpeg"
    }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

if files.isEmpty {
    fputs("No .jpg/.jpeg files found in: \(dirURL.path)\n", stderr)
    exit(1)
}

if jsonOutput {
    // JSON Lines (one JSON object per image)
    for url in files {
        autoreleasepool {
            do {
                let lines = try ocrImage(at: url, languages: languages)
                let obj: [String: Any] = [
                    "file": url.lastPathComponent,
                    "text": lines.joined(separator: "\n")
                ]
                let data = try JSONSerialization.data(withJSONObject: obj, options: [])
                if let s = String(data: data, encoding: .utf8) {
                    print(s)
                }
            } catch {
                fputs("Failed \(url.lastPathComponent): \(error)\n", stderr)
            }
        }
    }
} else {
    // Human-readable blocks
    for url in files {
        autoreleasepool {
            do {
                let lines = try ocrImage(at: url, languages: languages)
                print("=== \(url.lastPathComponent) ===")
                print(lines.joined(separator: "\n"))
                print("")
            } catch {
                fputs("Failed \(url.lastPathComponent): \(error)\n", stderr)
            }
        }
    }
}

