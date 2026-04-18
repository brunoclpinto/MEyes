import CoreImage
import ImageIO
import Foundation
import UniformTypeIdentifiers

/// Saves individual HEIC frames to the app's Documents directory with
/// timing metadata embedded in EXIF and a per-session JSON manifest.
///
/// Directory structure:
/// ```
/// Documents/HackEye/Session_yyyyMMdd_HHmmss/
///   frame_000001.heic
///   frame_000002.heic
///   session.json
/// ```
///
/// Each HEIC file carries timing data in:
/// - **EXIF UserComment** — machine-readable JSON
/// - **TIFF ImageDescription** — human-readable summary (visible in Finder Get Info)
final class DebugFrameSaver {

  // MARK: - State

  private let queue = DispatchQueue(label: "DebugFrameSaver.queue")
  private let ciContext = CIContext()

  private var sessionDir: URL?
  private var frameIndex: Int = 0
  private var manifest: [FrameEntry] = []
  private var startTime: CFAbsoluteTime = 0
  private var sessionStartDate: Date?

  // MARK: - Manifest Types

  private nonisolated struct FrameEntry: Encodable {
    let filename: String
    let elapsed: Double
    let fps: Double
    let flowTimings: [String: Double]
    let totalMs: Double
  }

  private nonisolated struct SessionManifest: Encodable {
    let sessionStart: String
    let frameCount: Int
    let frames: [FrameEntry]
  }

  // MARK: - Lifecycle

  func start() {
    queue.async {
      self.frameIndex = 0
      self.manifest = []
      self.startTime = 0
      self.sessionStartDate = Date()

      let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
      let hackEyeDir = docs.appendingPathComponent("HackEye", isDirectory: true)
      let ts = Self.timestampString(from: self.sessionStartDate!)
      let sessionDir = hackEyeDir.appendingPathComponent("Session_\(ts)", isDirectory: true)

      do {
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
      } catch {
        print("[DebugFrameSaver] Failed to create session directory: \(error)")
        return
      }

      self.sessionDir = sessionDir
      print("[DebugFrameSaver] Session started: \(sessionDir.path)")
    }
  }

  func appendFrame(
    _ frame: CIImage,
    timing: BusApproachTracker.TimingInfo?,
    currentFPS: Double
  ) {
    queue.async {
      guard let sessionDir = self.sessionDir else { return }

      let now = CFAbsoluteTimeGetCurrent()
      if self.startTime == 0 { self.startTime = now }
      let elapsed = now - self.startTime

      self.frameIndex += 1
      let filename = String(format: "frame_%06d.heic", self.frameIndex)
      let fileURL = sessionDir.appendingPathComponent(filename)

      // Build timing dictionaries
      var flowDict: [String: Double] = [:]
      if let timing {
        for (name, ms) in timing.flowTimings {
          flowDict[name] = ms
        }
      }

      let entry = FrameEntry(
        filename: filename,
        elapsed: elapsed,
        fps: currentFPS,
        flowTimings: flowDict,
        totalMs: timing?.totalMs ?? 0
      )

      // Render CIImage → CGImage
      guard let cgImage = self.ciContext.createCGImage(frame, from: frame.extent) else {
        print("[DebugFrameSaver] Failed to create CGImage for \(filename)")
        return
      }

      // Build EXIF metadata
      let properties = Self.buildImageProperties(
        fps: currentFPS,
        timing: timing,
        elapsed: elapsed
      )

      // Write HEIC with metadata
      guard let dest = CGImageDestinationCreateWithURL(
        fileURL as CFURL,
        UTType.heic.identifier as CFString,
        1,
        nil
      ) else {
        print("[DebugFrameSaver] Failed to create image destination for \(filename)")
        return
      }

      CGImageDestinationAddImage(dest, cgImage, properties as CFDictionary)

      if !CGImageDestinationFinalize(dest) {
        print("[DebugFrameSaver] Failed to finalize \(filename)")
        return
      }

      self.manifest.append(entry)
    }
  }

  func stop() {
    queue.async {
      guard let sessionDir = self.sessionDir else { return }

      // Write session.json manifest
      let iso = ISO8601DateFormatter()
      let sessionManifest = SessionManifest(
        sessionStart: iso.string(from: self.sessionStartDate ?? Date()),
        frameCount: self.manifest.count,
        frames: self.manifest
      )

      let manifestURL = sessionDir.appendingPathComponent("session.json")
      do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(sessionManifest)
        try data.write(to: manifestURL)
      } catch {
        print("[DebugFrameSaver] Failed to write session.json: \(error)")
      }

      print("[DebugFrameSaver] Session stopped: \(self.manifest.count) frames saved to \(sessionDir.path)")

      // Reset state
      self.sessionDir = nil
      self.frameIndex = 0
      self.manifest = []
      self.startTime = 0
      self.sessionStartDate = nil
    }
  }

  // MARK: - Private

  private static func buildImageProperties(
    fps: Double,
    timing: BusApproachTracker.TimingInfo?,
    elapsed: Double
  ) -> [CFString: Any] {
    // Machine-readable JSON for EXIF UserComment
    var jsonDict: [String: Any] = [
      "fps": fps,
      "elapsed": elapsed
    ]
    if let timing {
      var flowDict: [String: Double] = [:]
      for (name, ms) in timing.flowTimings {
        flowDict[name] = ms
      }
      jsonDict["flowTimings"] = flowDict
      jsonDict["totalMs"] = timing.totalMs
    }
    let jsonString: String
    if let data = try? JSONSerialization.data(withJSONObject: jsonDict, options: [.sortedKeys]),
       let str = String(data: data, encoding: .utf8) {
      jsonString = str
    } else {
      jsonString = "{}"
    }

    // Human-readable summary for TIFF ImageDescription
    var lines: [String] = []
    lines.append(String(format: "FPS: %.1f", fps))
    if let timing {
      for (name, ms) in timing.flowTimings {
        lines.append(String(format: "%@: %.1f ms", name, ms))
      }
      lines.append(String(format: "workflow: %.1f ms", timing.totalMs))
    }
    lines.append(String(format: "elapsed: %.3f s", elapsed))
    let humanReadable = lines.joined(separator: "\n")

    return [
      kCGImagePropertyExifDictionary: [
        kCGImagePropertyExifUserComment: jsonString
      ],
      kCGImagePropertyTIFFDictionary: [
        kCGImagePropertyTIFFImageDescription: humanReadable
      ]
    ]
  }

  private static func timestampString(from date: Date) -> String {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyyMMdd_HHmmss"
    return f.string(from: date)
  }
}
