//
//  Camera.view.model.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI
internal import Combine
import AVFoundation
import CoreGraphics

/// Describes what the camera view should display in its action area.
enum CameraAction {
  /// An interactive button the user can tap.
  case button(icon: String, label: String, hint: String)
  /// A non-interactive status message (with a spinner).
  case status(message: String)
}

@MainActor
class CameraViewModel: ObservableObject {
  @Published var camera: CameraSnapshot
  @Published var state: CameraState = .disconnected(.notInit)
  @Published var action: CameraAction = .status(message: "Initializing")
  
  private var stateTask: Task<Void, Never>?
  private let tracker: BusApproachTracker?
  private let speaker = Speaker()

  #if DevDebug
  private var recorder: CGImageVideoRecorder?
  private var lastFrameTime: CFAbsoluteTime = 0
  private var currentFPS: Double = 0
  private var lastTiming: BusApproachTracker.TimingInfo?
  #endif
  
  init(camera: CameraSnapshot) {
    self.camera = camera

    let stage1 = try? YOLOModel(.bundle(name: "yolo26sINT8512x896"))
    let stage2 = try? YOLOModel(.bundle(name: "busInfoYolo26sINT8512x896"))
    self.tracker = BusApproachTracker(stage1Model: stage1, stage2Model: stage2)
  }
  
  func processFrame(_ frame: CGImage) async {
    guard let tracker else { return }

    #if DevDebug
    // FPS calculation
    let now = CFAbsoluteTimeGetCurrent()
    if lastFrameTime > 0 {
      let delta = now - lastFrameTime
      if delta > 0 { currentFPS = 1.0 / delta }
    }
    lastFrameTime = now

    // Record the original frame before any processing
    try? recorder?.append(frame)
    #endif

    do {
      let (results, timing) = try await tracker.processFrame(frame)

      #if DevDebug
      lastTiming = timing
      #endif

      for bus in results {
        let number = bus.ocrText.leadingNaturalNumber()
        guard !number.isEmpty else { continue }
        speaker.speak(number)
      }
    } catch {
      print("[CameraViewModel] processFrame error: \(error)")
    }
  }
  
  func performAction() async {
    guard let device = camera.device else {
      switch state {
        case .disconnected(_):
          break
        default:
          state = .disconnected(.notInit)
      }
      return
    }
    
    switch state {
      case .connected, .stopped:
        #if DevDebug
        startRecording()
        #endif
        await device.start()
      case .started:
        await device.stop()
        #if DevDebug
        stopRecording()
        #endif
      default:
        break
    }
  }

  #if DevDebug
  // MARK: - Debug: Video Recording

  private func startRecording() {
    // Use a common frame size; recorder will skip mismatched frames
    let size = CGSize(width: 1280, height: 720)
    recorder = CGImageVideoRecorder(size: size)
    do {
      try recorder?.start()
      print("[DevDebug] Recording started")
    } catch {
      print("[DevDebug] Failed to start recording: \(error)")
      recorder = nil
    }
    lastFrameTime = 0
    currentFPS = 0
    lastTiming = nil
  }

  private func stopRecording() {
    recorder?.stopAndSaveToPhotos { result in
      switch result {
        case .success:
          print("[DevDebug] Video saved to Photos")
        case .failure(let error):
          print("[DevDebug] Failed to save video: \(error)")
      }
    }
    recorder = nil
  }

  // MARK: - Debug: Overlay

  func overlayFrame(_ frame: CGImage) -> CGImage {
    let w = frame.width
    let h = frame.height
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let ctx = CGContext(
      data: nil, width: w, height: h,
      bitsPerComponent: 8, bytesPerRow: 0,
      space: colorSpace, bitmapInfo: bitmapInfo
    ) else { return frame }

    // Draw the original frame
    ctx.draw(frame, in: CGRect(x: 0, y: 0, width: w, height: h))

    // Build overlay text lines
    var lines: [String] = []
    lines.append(String(format: "FPS: %.1f", currentFPS))

    if let timing = lastTiming {
      for (name, ms) in timing.flowTimings {
        lines.append(String(format: "%@: %.1f ms", name, ms))
      }
      lines.append(String(format: "workflow: %.1f ms", timing.totalMs))
    }

    // Text rendering setup — use bottom-left quarter as dead area
    let fontSize = CGFloat(h) / 18  // large enough to fill the quarter
    let lineHeight = fontSize * 1.3
    let margin = CGFloat(h) * 0.02

    // Semi-transparent background behind text
    let textBlockHeight = lineHeight * CGFloat(lines.count) + margin * 2
    let textBlockWidth = CGFloat(w) / 2
    ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 0.6)
    ctx.fill(CGRect(x: 0, y: 0, width: textBlockWidth, height: textBlockHeight))

    // Draw text lines (CGContext y=0 is bottom)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: UIFont.monospacedSystemFont(ofSize: fontSize, weight: .bold),
      .foregroundColor: UIColor.green
    ]

    UIGraphicsPushContext(ctx)

    // Flip context for UIKit text drawing
    ctx.saveGState()
    ctx.textMatrix = .identity
    ctx.translateBy(x: 0, y: CGFloat(h))
    ctx.scaleBy(x: 1, y: -1)

    // Text starts at bottom-left: in flipped coords that's near y = h - margin
    let startY = CGFloat(h) - textBlockHeight + margin

    for (i, line) in lines.enumerated() {
      let y = startY + CGFloat(i) * lineHeight
      let str = NSAttributedString(string: line, attributes: attrs)
      str.draw(at: CGPoint(x: margin, y: y))
    }

    ctx.restoreGState()
    UIGraphicsPopContext()

    return ctx.makeImage() ?? frame
  }
  #endif
  
  func startObservingState() async {
    guard let camera = self.camera.device else { return }
    stateTask?.cancel()
    stateTask = Task { [weak self] in
      guard let self else { return }
      for await state in await camera.stateUpdates() {
        self.state = state
        self.action = self.actionForState(state)
      }
    }
  }

  func stopObservingState() {
    stateTask?.cancel()
    stateTask = nil
  }

  // MARK: - Private
  
  private func actionForState(_ state: CameraState) -> CameraAction {
    if camera.isRegistration {
      return actionForRegistrationState(state)
    }
    return actionForCameraState(state)
  }

  private func actionForCameraState(_ state: CameraState) -> CameraAction {
    switch state {
      case .connected, .stopped:
        return .button(
          icon: "play.fill",
          label: String(localized: "Start"),
          hint: String(localized: "Start using camera")
        )
      case .started:
        return .button(
          icon: "stop.fill",
          label: String(localized: "Stop"),
          hint: String(localized: "Stop using camera")
        )
      case .connecting:
        return .status(message: String(localized: "Connecting to camera"))
      case .starting:
        return .status(message: String(localized: "Starting camera feed"))
      case .stopping:
        return .status(message: String(localized: "Stopping camera feed"))
      case .disconnecting:
        return .status(message: String(localized: "Disconnecting from camera"))
      case .disconnected(let error):
        if let error {
          return .status(message: error.rawValue)
        }
        return .status(message: String(localized: "Disconnected"))
      case .forceDisconnect:
        return .status(message: String(localized: "Connection lost"))
    }
  }

  private func actionForRegistrationState(_ state: CameraState) -> CameraAction {
    switch state {
      case .connected:
        return .button(
          icon: "link.badge.plus",
          label: String(localized: "Register"),
          hint: String(localized: "Opens Meta AI to register this app with your glasses")
        )
      case .connecting, .starting:
        return .status(
          message: String(localized: "Waiting for registration in Meta AI. Approve the request and return to this app.")
        )
      case .started:
        return .status(
          message: String(localized: "Registration complete. Discovering cameras.")
        )
      default:
        return .status(message: state.stringValue)
    }
  }
}
