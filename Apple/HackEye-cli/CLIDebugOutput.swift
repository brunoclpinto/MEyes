import Foundation
import CoreImage
import CoreGraphics
import ImageIO

// MARK: - CLIDebugOutput

final class CLIDebugOutput {
    let basePath: String
    private let ciContext: CIContext

    init(basePath: String) {
        self.basePath = basePath
        self.ciContext = ImageLetterboxer.ciContext
    }

    /// Resolve a unique output folder, Finder-style: "name", "name 2", "name 3", ...
    func resolveOutputFolder(inputName: String) -> URL {
        let base = URL(fileURLWithPath: basePath)
        var candidate = base.appendingPathComponent(inputName)
        if !FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }
        var n = 2
        while FileManager.default.fileExists(
            atPath: base.appendingPathComponent("\(inputName) \(n)").path
        ) {
            n += 1
        }
        candidate = base.appendingPathComponent("\(inputName) \(n)")
        return candidate
    }

    func saveFrame(
        folder: URL,
        frameIndex: Int,
        originalImage: CIImage,
        meta: LetterboxMeta,
        detections: [BusDetection],
        tracked: [TrackedBus],
        busInfoResults: [(TrackedBus, BusInfoResult, CIImage?)],
        fps: Double?,
        s1Ms: Double, s2Ms: Double, s3Ms: Double, totalMs: Double
    ) throws {
        let frameFolder = folder.appendingPathComponent(
            String(format: "frame_%04d", frameIndex)
        )
        try FileManager.default.createDirectory(
            at: frameFolder, withIntermediateDirectories: true
        )

        // original.png
        saveCIImageAsPNG(originalImage, to: frameFolder.appendingPathComponent("original.png"))

        // info.json
        var info: [String: Any] = [
            "frameIndex": frameIndex,
            "totalProcessingMs": totalMs,
            "width": Int(originalImage.extent.width),
            "height": Int(originalImage.extent.height)
        ]
        if let fps = fps { info["fps"] = fps }
        saveJSON(info, to: frameFolder.appendingPathComponent("info.json"))

        // BusDetection/
        let detFolder = frameFolder.appendingPathComponent("BusDetection")
        try FileManager.default.createDirectory(at: detFolder, withIntermediateDirectories: true)

        // work.png — recreate letterboxed frame
        let (letterboxed, _) = ImageLetterboxer.letterboxWithMeta(
            originalImage,
            srcW: meta.srcW, srcH: meta.srcH,
            dstW: meta.dstW, dstH: meta.dstH
        )
        saveCIImageAsPNG(letterboxed, to: detFolder.appendingPathComponent("work.png"))

        let detResult: [String: Any] = [
            "detected": !detections.isEmpty,
            "count": detections.count,
            "elapsedMs": s1Ms,
            "detections": detections.map { d in
                [
                    "score": d.score,
                    "class": d.cls,
                    "boxDetector": [
                        "x1": d.boxDetector.x1, "y1": d.boxDetector.y1,
                        "x2": d.boxDetector.x2, "y2": d.boxDetector.y2
                    ],
                    "boxOriginal": [
                        "x1": d.boxOriginal.x1, "y1": d.boxOriginal.y1,
                        "x2": d.boxOriginal.x2, "y2": d.boxOriginal.y2
                    ]
                ] as [String: Any]
            }
        ]
        saveJSON(detResult, to: detFolder.appendingPathComponent("result.json"))

        // BusTracking/
        let trackFolder = frameFolder.appendingPathComponent("BusTracking")
        try FileManager.default.createDirectory(at: trackFolder, withIntermediateDirectories: true)

        let trackResult: [String: Any] = [
            "detected": !tracked.isEmpty,
            "count": tracked.count,
            "elapsedMs": s2Ms,
            "buses": tracked.map { t in
                [
                    "id": t.id,
                    "name": t.name,
                    "isApproaching": t.isApproaching,
                    "approachingScore": t.approachingScore,
                    "score": t.lastScore
                ] as [String: Any]
            }
        ]
        saveJSON(trackResult, to: trackFolder.appendingPathComponent("result.json"))

        // BusInfo/
        if !busInfoResults.isEmpty {
            let infoFolder = frameFolder.appendingPathComponent("BusInfo")
            try FileManager.default.createDirectory(at: infoFolder, withIntermediateDirectories: true)

            for (i, (bus, result, cropImage)) in busInfoResults.enumerated() {
                let busFolder = infoFolder.appendingPathComponent("bus_\(i)")
                try FileManager.default.createDirectory(at: busFolder, withIntermediateDirectories: true)

                // work.png — the cropped bus region
                if let crop = cropImage {
                    saveCIImageAsPNG(crop, to: busFolder.appendingPathComponent("work.png"))
                }

                let busResult: [String: Any] = [
                    "busId": bus.id,
                    "busName": bus.name,
                    "ocrRaw": result.ocrText,
                    "ocrSpoken": result.ocrText.leadingNaturalNumber(),
                    "infoBox": result.infoBoxOriginal.map { box in
                        ["x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2] as [String: Any]
                    } as Any
                ]
                saveJSON(busResult, to: busFolder.appendingPathComponent("result.json"))
            }
        }
    }

    // MARK: - Image & JSON Helpers

    private func saveCIImageAsPNG(_ image: CIImage, to url: URL) {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else { return }
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else { return }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }

    private func saveJSON(_ dict: [String: Any], to url: URL) {
        if let data = try? JSONSerialization.data(
            withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]
        ) {
            try? data.write(to: url)
        }
    }
}
