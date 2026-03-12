import Foundation
import CoreImage
import CoreML
import AVFoundation

// MARK: - Configuration

struct CLIConfig {
    let inputPath: String
    let stage1ModelPath: String
    let stage2ModelPath: String
    let detectorW: Int
    let detectorH: Int
    let stage1Conf: Double
    let stage2Conf: Double
    let stage1BusClass: Int
    let stepsEnabled: Bool
    let debugDestination: String?
    let verbose: Bool
    let jsonOutput: Bool
}

// MARK: - Argument Parsing

enum CLIError: Error, CustomStringConvertible {
    case missingRequired(String)
    case invalidValue(key: String, value: String, expected: String)
    case stepsRequiresDebugDestination
    case inputNotFound(String)
    case noImagesInFolder(String)
    case cannotLoadImage(String)
    case noVideoTrack(String)
    case modelCompileFailed(String, String)
    case unknownArgument(String)

    var description: String {
        switch self {
        case .missingRequired(let k): return "Missing required argument: \(k)=<value>"
        case .invalidValue(let k, let v, let t): return "Invalid value for \(k)=\(v), expected \(t)"
        case .stepsRequiresDebugDestination: return "steps=true requires debugDestination=<path>"
        case .inputNotFound(let p): return "Input not found: \(p)"
        case .noImagesInFolder(let p): return "No image files found in: \(p)"
        case .cannotLoadImage(let p): return "Cannot load image: \(p)"
        case .noVideoTrack(let p): return "No video track in: \(p)"
        case .modelCompileFailed(let n, let p): return "\(n) model failed to compile: \(p)"
        case .unknownArgument(let a): return "Unknown argument: \(a)"
        }
    }
}

func parseArguments() throws -> CLIConfig {
    let args = CommandLine.arguments.dropFirst()
    let knownKeys: Set<String> = [
        "input", "stage1Model", "stage2Model",
        "detectorW", "detectorH", "stage1Conf", "stage2Conf", "stage1BusClass",
        "steps", "debugDestination", "verbose", "json"
    ]

    var dict: [String: String] = [:]
    for arg in args {
        guard let eqIdx = arg.firstIndex(of: "=") else {
            throw CLIError.unknownArgument(arg)
        }
        let key = String(arg[arg.startIndex..<eqIdx])
        let value = String(arg[arg.index(after: eqIdx)...])
        if !knownKeys.contains(key) {
            stderr("Warning: unknown argument '\(key)', ignoring")
        }
        dict[key] = value
    }

    guard let inputPath = dict["input"] else {
        throw CLIError.missingRequired("input")
    }
    guard let s1Path = dict["stage1Model"] else {
        throw CLIError.missingRequired("stage1Model")
    }
    guard let s2Path = dict["stage2Model"] else {
        throw CLIError.missingRequired("stage2Model")
    }

    func intVal(_ key: String, default d: Int) throws -> Int {
        guard let s = dict[key] else { return d }
        guard let v = Int(s) else { throw CLIError.invalidValue(key: key, value: s, expected: "integer") }
        return v
    }
    func doubleVal(_ key: String, default d: Double) throws -> Double {
        guard let s = dict[key] else { return d }
        guard let v = Double(s) else { throw CLIError.invalidValue(key: key, value: s, expected: "decimal") }
        return v
    }

    let stepsEnabled = dict["steps"]?.lowercased() == "true"
    let debugDest = dict["debugDestination"]
    if stepsEnabled && debugDest == nil {
        throw CLIError.stepsRequiresDebugDestination
    }
    let verbose = dict["verbose"]?.lowercased() == "true"
    let jsonOutput = dict["json"]?.lowercased() == "true"

    return CLIConfig(
        inputPath: inputPath,
        stage1ModelPath: s1Path,
        stage2ModelPath: s2Path,
        detectorW: try intVal("detectorW", default: 512),
        detectorH: try intVal("detectorH", default: 896),
        stage1Conf: try doubleVal("stage1Conf", default: 0.51),
        stage2Conf: try doubleVal("stage2Conf", default: 0.50),
        stage1BusClass: try intVal("stage1BusClass", default: 5),
        stepsEnabled: stepsEnabled,
        debugDestination: debugDest,
        verbose: verbose,
        jsonOutput: jsonOutput
    )
}

// MARK: - Input Feed

private let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tiff", "tif", "bmp"]
private let videoExtensions: Set<String> = ["mov", "mp4", "m4v", "avi"]

enum InputType {
    case singleImage(URL)
    case video(URL)
    case imageFolder(URL, files: [URL])
}

func classifyInput(_ path: String) throws -> InputType {
    let url = URL(fileURLWithPath: path)
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else {
        throw CLIError.inputNotFound(path)
    }
    if isDir.boolValue {
        let contents = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        let images = contents
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !images.isEmpty else { throw CLIError.noImagesInFolder(path) }
        return .imageFolder(url, files: images)
    }
    let ext = url.pathExtension.lowercased()
    if videoExtensions.contains(ext) {
        return .video(url)
    }
    return .singleImage(url)
}

func loadImage(from url: URL) throws -> CIImage {
    guard let ci = CIImage(contentsOf: url) else {
        throw CLIError.cannotLoadImage(url.path)
    }
    return ci.transformed(by: CGAffineTransform(
        translationX: -ci.extent.origin.x, y: -ci.extent.origin.y
    ))
}

// MARK: - Model Loading

func compileAndLoadModel(name: String, path: String) throws -> YOLOModel {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path) else {
        throw CLIError.inputNotFound(path)
    }
    let compiledURL = try MLModel.compileModel(at: url)
    return try YOLOModel(.compiledURL(compiledURL))
}

// MARK: - Helpers

func stderr(_ message: String) {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
}

// MARK: - Entry Point

let semaphore = DispatchSemaphore(value: 0)
var exitCode: Int32 = 0

Task {
    do {
        let config = try parseArguments()
        stderr("Compiling stage1 model...")
        let s1 = try compileAndLoadModel(name: "stage1", path: config.stage1ModelPath)
        stderr("Compiling stage2 model...")
        let s2 = try compileAndLoadModel(name: "stage2", path: config.stage2ModelPath)
        let processor = CLIProcessor(config: config, stage1: s1, stage2: s2)
        try await processor.run()
    } catch {
        stderr("Error: \(error)")
        exitCode = 1
    }
    semaphore.signal()
}

semaphore.wait()
exit(exitCode)
