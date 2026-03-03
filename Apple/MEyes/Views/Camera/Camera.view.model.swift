//
//  Camera.view.model.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI
internal import Combine
import AVFoundation

enum ActionButtonState: String {
  case play
  case stop
  
  var rawValue: String {
    switch self {
      case .play:
        return "play.fill"
      case .stop:
        return "stop.fill"
    }
  }
  
  var accessibleTitle: String {
    switch self {
      case .play:
        return String(localized: "Start")
      case .stop:
        return String(localized: "Stop")
    }
  }
  
  var accessibleHint: String {
    switch self {
      case .play:
        return String(localized: "Start using camera")
      case .stop:
        return String(localized: "Stop using camera")
    }
  }
}

@MainActor
class CameraViewModel: ObservableObject {
  @Published var camera: CameraSnapshot
  @Published var state: CameraState = .disconnected(.notInit)
  @Published var actionButtonIcon: ActionButtonState = .play
  @Published var actionButtonEnabled: Bool = false
  
  private var stateTask: Task<Void, Never>?
  
  init(camera: CameraSnapshot) {
    self.camera = camera
  }
  
  func performActionButton() async {
    guard let device = camera.device else {
      switch state {
        case .disconnected(_):
          break
        default:
          state = .disconnected(.notInit)
      }
      return
    }
    switch actionButtonIcon {
      case .play:
        await device.start()
      case .stop:
        await device.stop()
    }
  }
  
  func startObservingState() async {
    guard let camera = self.camera.device else { return }
    stateTask?.cancel()
    stateTask = Task { [weak self] in
      guard let self else {return}
      for await state in await camera.stateUpdates() {
        self.state = state
        switch state {
          case
              .connecting,
              .disconnecting,
              .starting,
              .stopping,
              .disconnected(_),
              .forceDisconnect,
              .stopped,
              .connected:
            self.actionButtonIcon = .play
          case
              .started:
            self.actionButtonIcon = .stop
        }
        switch state {
          case
              .connecting,
              .disconnecting,
              .starting,
              .stopping,
              .disconnected(_),
              .forceDisconnect:
            self.actionButtonEnabled = false
            case
              .stopped,
              .connected,
              .started:
            self.actionButtonEnabled = true
        }
      }
    }
  }

  func stopObservingState() {
    stateTask?.cancel()
    stateTask = nil
  }
}
