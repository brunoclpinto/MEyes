import Foundation
import CoreImage
import CoreGraphics
import AVFoundation
import Vision

// MARK: - JSON Output Types

struct CLIResult: Encodable {
    let input: String
    let frames: [CLIFrameResult]
}

struct CLIFrameResult: Encodable {
    let filePath: String
    let frameNumber: Int
    let BusDetection: CLIDetectionResult
    let BusTracking: CLITrackingResult
    let BusInfo: CLIInfoResult
}

struct CLIDetectionResult: Encodable {
    let detected: Bool
    let count: Int
}

struct CLITrackingResult: Encodable {
    let detected: Bool
    let count: Int
}

struct CLIInfoResult: Encodable {
    let detected: Bool
    let results: [CLIBusInfoEntry]
}

struct CLIBusInfoEntry: Encodable {
    let ocrRaw: String
    let ocrSpoken: String
}

// MARK: - CLIProcessor

final class CLIProcessor {
    let config: CLIConfig
    private let busDetector: BusDetector
    private let busTracker: BusTracker
    private let busInfoDetector: BusInfoDetector
    private let debugOutput: CLIDebugOutput?

    init(config: CLIConfig, stage1: YOLOModel, stage2: YOLOModel) {
        self.config = config

        var d1cfg = BusDetector.Config()
        d1cfg.detectorW = config.detectorW
        d1cfg.detectorH = config.detectorH
        d1cfg.busClass = config.stage1BusClass
        d1cfg.confidence = config.stage1Conf

        var d2cfg = BusInfoDetector.Config()
        d2cfg.detectorW = config.detectorW
        d2cfg.detectorH = config.detectorH
        d2cfg.confidence = config.stage2Conf

        busDetector = BusDetector(model: stage1, config: d1cfg)
        busTracker = BusTracker()
        busInfoDetector = BusInfoDetector(model: stage2, config: d2cfg)

        if config.stepsEnabled, let dest = config.debugDestination {
            debugOutput = CLIDebugOutput(basePath: dest)
        } else {
            debugOutput = nil
        }
    }

    func run() async throws {
        let inputType = try classifyInput(config.inputPath)

        switch inputType {
        case .singleImage(let url):
            let result = try await processSingleImage(url)
            if config.jsonOutput { printJSON(result) }

        case .video(let url):
            let result = try await processVideo(url)
            if config.jsonOutput { printJSON(result) }

        case .imageFolder(_, let files):
            for file in files {
                let result = try await processSingleImage(file)
                if config.jsonOutput { printJSON(result) }
            }
        }
    }

    // MARK: - Single Image

    private func processSingleImage(_ url: URL) async throws -> CLIResult {
        let image = try loadImage(from: url)
        let inputName = url.deletingPathExtension().lastPathComponent
        let outputFolder = debugOutput?.resolveOutputFolder(inputName: inputName)
        let frameResult = try await processFrame(
            image, index: 0, filePath: url.path, outputFolder: outputFolder
        )
        return CLIResult(input: url.path, frames: [frameResult])
    }

    // MARK: - Video

    private func processVideo(_ url: URL) async throws -> CLIResult {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else {
            throw CLIError.noVideoTrack(url.path)
        }

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        let fps = try await Double(track.load(.nominalFrameRate))
        let inputName = url.deletingPathExtension().lastPathComponent
        let outputFolder = debugOutput?.resolveOutputFolder(inputName: inputName)

        var frames: [CLIFrameResult] = []
        var index = 0

        while reader.status == .reading,
              let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let ci = CIImage(cvPixelBuffer: pixelBuffer)
                .transformed(by: CGAffineTransform(
                    translationX: -CIImage(cvPixelBuffer: pixelBuffer).extent.origin.x,
                    y: -CIImage(cvPixelBuffer: pixelBuffer).extent.origin.y
                ))

            let frameResult = try await processFrame(
                ci, index: index, filePath: url.path, outputFolder: outputFolder, fps: fps
            )
            frames.append(frameResult)
            index += 1
        }

        stderr("Processed \(index) frames total")
        return CLIResult(input: url.path, frames: frames)
    }

    // MARK: - Per-Frame Pipeline

    private func processFrame(
        _ image: CIImage,
        index: Int,
        filePath: String,
        outputFolder: URL?,
        fps: Double? = nil
    ) async throws -> CLIFrameResult {
        let frameStart = CFAbsoluteTimeGetCurrent()
        let srcW = Double(image.extent.width)
        let srcH = Double(image.extent.height)

        // Stage 1: Detection
        let s1Start = CFAbsoluteTimeGetCurrent()
        let (detections, meta) = try busDetector.detect(frame: image)
        let s1Ms = (CFAbsoluteTimeGetCurrent() - s1Start) * 1000

        // Stage 2: Tracking
        let s2Start = CFAbsoluteTimeGetCurrent()
        let tracked = busTracker.update(detections: detections)
        let s2Ms = (CFAbsoluteTimeGetCurrent() - s2Start) * 1000

        // Stage 3: Info for ALL tracked buses
        let s3Start = CFAbsoluteTimeGetCurrent()
        var busInfoResults: [(TrackedBus, BusInfoResult, CIImage?)] = []

        for bus in tracked {
            guard let det = detections.first(where: {
                iou($0.boxDetector, bus.lastBoxDetector) >= 0.5
            }) else { continue }

            if det.boxOriginal.w < 2 || det.boxOriginal.h < 2 { continue }

            let busCropCI = cropFromTopLeftAndNormalize(
                image, srcW: srcW, srcH: srcH, boxTopLeft: det.boxOriginal
            )
            let bw = Double(busCropCI.extent.width)
            let bh = Double(busCropCI.extent.height)
            guard bw >= 2, bh >= 2 else { continue }

            let infoResult = try await busInfoDetector.detect(
                busCropCI: busCropCI, busCropW: bw, busCropH: bh,
                ocrPreset: .default,
                recognitionLanguages: ["pt-PT"],
                usesLanguageCorrection: false,
                recognitionLevel: .accurate
            )

            busInfoResults.append((bus, infoResult, busCropCI))
        }
        let s3Ms = (CFAbsoluteTimeGetCurrent() - s3Start) * 1000
        let totalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000

        // Debug output
        if let folder = outputFolder {
            try debugOutput?.saveFrame(
                folder: folder,
                frameIndex: index,
                originalImage: image,
                meta: meta,
                detections: detections,
                tracked: tracked,
                busInfoResults: busInfoResults,
                fps: fps,
                s1Ms: s1Ms, s2Ms: s2Ms, s3Ms: s3Ms, totalMs: totalMs
            )
        }

        // Build JSON result
        let infoEntries: [CLIBusInfoEntry] = busInfoResults.map { (_, info, _) in
            CLIBusInfoEntry(
                ocrRaw: info.ocrText,
                ocrSpoken: info.ocrText.leadingNaturalNumber()
            )
        }

        let result = CLIFrameResult(
            filePath: filePath,
            frameNumber: index,
            BusDetection: CLIDetectionResult(
                detected: !detections.isEmpty,
                count: detections.count
            ),
            BusTracking: CLITrackingResult(
                detected: !tracked.isEmpty,
                count: tracked.count
            ),
            BusInfo: CLIInfoResult(
                detected: !infoEntries.isEmpty,
                results: infoEntries
            )
        )

        // Verbose output to stderr
        if config.verbose {
            var parts: [String] = []
            parts.append("[\(filePath)] frame \(index)")
            parts.append("detection: \(detections.count) bus(es)")
            parts.append("tracking: \(tracked.count) tracked")
            if !infoEntries.isEmpty {
                let ocrSummary = infoEntries.map { entry in
                    entry.ocrSpoken.isEmpty ? entry.ocrRaw : entry.ocrSpoken
                }.joined(separator: ", ")
                parts.append("OCR: \(ocrSummary)")
            }
            parts.append(String(format: "%.0f ms", totalMs))
            stderr(parts.joined(separator: " | "))
        }

        return result
    }

    // MARK: - Helpers

    private func iou(_ a: Box, _ b: Box) -> Double {
        let xA = max(a.x1, b.x1), yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2), yB = min(a.y2, b.y2)
        let inter = max(0, xB - xA) * max(0, yB - yA)
        let union = a.area + b.area - inter
        if union <= 0 { return 0 }
        return inter / union
    }

    private func cropFromTopLeftAndNormalize(
        _ src: CIImage, srcW: Double, srcH: Double, boxTopLeft: Box
    ) -> CIImage {
        let ciY = srcH - (boxTopLeft.y1 + boxTopLeft.h)
        let cropRect = CGRect(
            x: boxTopLeft.x1, y: ciY,
            width: boxTopLeft.w, height: boxTopLeft.h
        ).integral
        let bounds = CGRect(x: 0, y: 0, width: srcW, height: srcH)
        let rr = cropRect.intersection(bounds).integral
        if rr.isNull || rr.width < 1 || rr.height < 1 { return CIImage.empty() }
        let cropped = src.cropped(to: rr)
        return cropped.transformed(by: CGAffineTransform(
            translationX: -rr.origin.x, y: -rr.origin.y
        ))
    }

    private func printJSON(_ value: some Encodable) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(value),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
}
